# flake8: noqa
"""
HAT training entry point.

Extends basicsr's training pipeline to support multiple simultaneous
training datasets (train_1, train_2, ...) and multiple validation
datasets (val_1, val_2, ...) declared in the YAML config.

Multi-dataset behaviour:
  - All datasets whose phase key starts with 'train' (e.g. train_1, train_2)
    are concatenated into a single ConcatDataset before building the
    DataLoader.  DataLoader settings (batch_size, num_workers, etc.) are
    taken from the *first* train dataset entry.
  - All datasets whose phase key starts with 'val' already worked in
    basicsr (separate val_loaders list); this is preserved as-is.
"""

import datetime
import logging
import math
import os.path as osp
import time

import torch
from torch.utils.data import ConcatDataset

import hat.archs
import hat.data
import hat.models

from basicsr.data import build_dataloader, build_dataset
from basicsr.data.data_sampler import EnlargedSampler
from basicsr.data.prefetch_dataloader import CPUPrefetcher, CUDAPrefetcher
from basicsr.models import build_model
from basicsr.utils import (AvgTimer, MessageLogger, check_resume,
                            get_env_info, get_root_logger, get_time_str,
                            init_tb_logger, init_wandb_logger,
                            make_exp_dirs, mkdir_and_rename, scandir)
from basicsr.utils.options import copy_opt_file, dict2str, parse_options


# ──────────────────────────────────────────────────────────────────────────────
# Logger initialisation (unchanged from basicsr)
# ──────────────────────────────────────────────────────────────────────────────

def _init_tb_loggers(opt):
    if (opt['logger'].get('wandb') is not None
            and opt['logger']['wandb'].get('project') is not None
            and 'debug' not in opt['name']):
        assert opt['logger'].get('use_tb_logger') is True, (
            'should turn on tensorboard when using wandb')
        init_wandb_logger(opt)
    tb_logger = None
    if opt['logger'].get('use_tb_logger') and 'debug' not in opt['name']:
        # Store TB logs inside the experiment folder for easy co-location.
        tb_logger = init_tb_logger(
            log_dir=osp.join(opt['path']['experiments_root'], 'tb_logger'))
    return tb_logger


# ──────────────────────────────────────────────────────────────────────────────
# Multi-dataset aware dataloader factory
# ──────────────────────────────────────────────────────────────────────────────

def _create_train_val_dataloader(opt, logger):
    """Build train and validation DataLoaders from the YAML datasets section.

    Supports any number of training datasets:
      train / train_1 / train_2 / train_3 / ...
    All are concatenated via ``torch.utils.data.ConcatDataset``.
    DataLoader hyper-parameters (batch_size, num_workers, prefetch_mode …)
    are taken from the **first** train dataset entry in YAML order.

    Validation datasets:
      val / val_1 / val_2 / ...
    Each becomes a separate DataLoader (standard basicsr behaviour).
    """
    train_sets = []
    train_opts = []   # dataset-level configs for each train split
    val_loaders = []

    for phase, dataset_opt in opt['datasets'].items():
        top = phase.split('_')[0]   # 'train_1' -> 'train',  'val_2' -> 'val'

        if top == 'train':
            dataset_opt['phase'] = 'train'
            ds = build_dataset(dataset_opt)
            train_sets.append(ds)
            train_opts.append(dataset_opt)
            logger.info(
                f'Train dataset [{dataset_opt["name"]}]: {len(ds)} images.')

        elif top == 'val':
            dataset_opt['phase'] = 'val'
            val_set = build_dataset(dataset_opt)
            val_loader = build_dataloader(
                val_set, dataset_opt,
                num_gpu=opt['num_gpu'],
                dist=opt['dist'],
                sampler=None,
                seed=opt['manual_seed'])
            logger.info(
                f'Val dataset [{dataset_opt["name"]}]: {len(val_set)} images.')
            val_loaders.append(val_loader)

        else:
            raise ValueError(
                f'Dataset phase "{phase}" is not recognized. '
                'Keys must start with "train" or "val".')

    if not train_sets:
        raise ValueError('No training dataset found in the YAML config.')

    # ---- Concatenate all train datasets ------------------------------------
    if len(train_sets) == 1:
        train_set = train_sets[0]
    else:
        train_set = ConcatDataset(train_sets)
        sizes = ', '.join(str(len(s)) for s in train_sets)
        logger.info(
            f'Concatenated {len(train_sets)} train datasets '
            f'(sizes: {sizes}) -> {len(train_set)} total images.')

    # ---- Build single DataLoader from the combined dataset -----------------
    # Loader hyper-params come from the first train entry.
    loader_opt = train_opts[0]
    dataset_enlarge_ratio = loader_opt.get('dataset_enlarge_ratio', 1)

    train_sampler = EnlargedSampler(
        train_set, opt['world_size'], opt['rank'], dataset_enlarge_ratio)
    train_loader = build_dataloader(
        train_set, loader_opt,
        num_gpu=opt['num_gpu'],
        dist=opt['dist'],
        sampler=train_sampler,
        seed=opt['manual_seed'])

    num_iter_per_epoch = math.ceil(
        len(train_set) * dataset_enlarge_ratio
        / (loader_opt['batch_size_per_gpu'] * opt['world_size']))
    total_iters = int(opt['train']['total_iter'])
    total_epochs = math.ceil(total_iters / num_iter_per_epoch)

    logger.info(
        'Training statistics:'
        f'\n\tTrain datasets      : {len(train_sets)}'
        f'\n\tTotal train images  : {len(train_set)}'
        f'\n\tEnlarge ratio       : {dataset_enlarge_ratio}'
        f'\n\tBatch size / gpu    : {loader_opt["batch_size_per_gpu"]}'
        f'\n\tWorld size (# gpus) : {opt["world_size"]}'
        f'\n\tIters / epoch       : {num_iter_per_epoch}'
        f'\n\tTotal epochs        : {total_epochs}; iters: {total_iters}.')

    return train_loader, train_sampler, val_loaders, total_epochs, total_iters


# ──────────────────────────────────────────────────────────────────────────────
# Resume-state loader (unchanged from basicsr)
# ──────────────────────────────────────────────────────────────────────────────

def _load_resume_state(opt):
    resume_state_path = None
    if opt['auto_resume']:
        state_path = osp.join('experiments', opt['name'], 'training_states')
        if osp.isdir(state_path):
            states = list(
                scandir(state_path, suffix='state', recursive=False,
                        full_path=False))
            if states:
                states = [float(v.split('.state')[0]) for v in states]
                resume_state_path = osp.join(
                    state_path, f'{max(states):.0f}.state')
                opt['path']['resume_state'] = resume_state_path
    else:
        if opt['path'].get('resume_state'):
            resume_state_path = opt['path']['resume_state']

    if resume_state_path is None:
        return None
    device_id = torch.cuda.current_device()
    resume_state = torch.load(
        resume_state_path,
        map_location=lambda storage, loc: storage.cuda(device_id))
    check_resume(opt, resume_state['iter'])
    return resume_state


# ──────────────────────────────────────────────────────────────────────────────
# Main training pipeline
# ──────────────────────────────────────────────────────────────────────────────

def train_pipeline(root_path):
    opt, args = parse_options(root_path, is_train=True)
    opt['root_path'] = root_path

    torch.backends.cudnn.benchmark = True

    resume_state = _load_resume_state(opt)

    if resume_state is None:
        make_exp_dirs(opt)
        if (opt['logger'].get('use_tb_logger')
                and 'debug' not in opt['name']
                and opt['rank'] == 0):
            mkdir_and_rename(
                osp.join(opt['path']['experiments_root'], 'tb_logger'))

    copy_opt_file(args.opt, opt['path']['experiments_root'])

    log_file = osp.join(
        opt['path']['log'],
        f"train_{opt['name']}_{get_time_str()}.log")
    logger = get_root_logger(
        logger_name='basicsr', log_level=logging.INFO, log_file=log_file)
    logger.info(get_env_info())
    logger.info(dict2str(opt))
    tb_logger = _init_tb_loggers(opt)

    # ---- Create dataloaders (multi-dataset aware) --------------------------
    result = _create_train_val_dataloader(opt, logger)
    train_loader, train_sampler, val_loaders, total_epochs, total_iters = result

    # ---- Build model -------------------------------------------------------
    model = build_model(opt)
    if resume_state:
        model.resume_training(resume_state)
        logger.info(
            f"Resuming training from epoch: {resume_state['epoch']}, "
            f"iter: {resume_state['iter']}.")
        start_epoch = resume_state['epoch']
        current_iter = resume_state['iter']
    else:
        start_epoch = 0
        current_iter = 0

    msg_logger = MessageLogger(opt, current_iter, tb_logger)

    # ---- Prefetcher --------------------------------------------------------
    # Use first train dataset's prefetch_mode setting.
    first_train_opt = next(
        v for k, v in opt['datasets'].items()
        if k.split('_')[0] == 'train')
    prefetch_mode = first_train_opt.get('prefetch_mode')

    if prefetch_mode is None or prefetch_mode == 'cpu':
        prefetcher = CPUPrefetcher(train_loader)
    elif prefetch_mode == 'cuda':
        prefetcher = CUDAPrefetcher(train_loader, opt)
        logger.info(f'Use {prefetch_mode} prefetch dataloader')
        if first_train_opt.get('pin_memory') is not True:
            raise ValueError(
                'Please set pin_memory=True for CUDAPrefetcher.')
    else:
        raise ValueError(
            f'Wrong prefetch_mode {prefetch_mode}. '
            "Supported ones are: None, 'cuda', 'cpu'.")

    # ---- Training loop -----------------------------------------------------
    logger.info(
        f'Start training from epoch: {start_epoch}, iter: {current_iter}')
    data_timer, iter_timer = AvgTimer(), AvgTimer()
    start_time = time.time()

    for epoch in range(start_epoch, total_epochs + 1):
        train_sampler.set_epoch(epoch)
        prefetcher.reset()
        train_data = prefetcher.next()

        while train_data is not None:
            data_timer.record()
            current_iter += 1
            if current_iter > total_iters:
                break

            model.update_learning_rate(
                current_iter,
                warmup_iter=opt['train'].get('warmup_iter', -1))
            model.feed_data(train_data)
            model.optimize_parameters(current_iter)
            iter_timer.record()

            if current_iter == 1:
                msg_logger.reset_start_time()

            # Log training sample visuals to TensorBoard
            train_vis_freq = opt['logger'].get('tb_train_vis_freq', 500)
            if (tb_logger is not None
                    and current_iter % train_vis_freq == 0):
                model.log_train_visuals(tb_logger, current_iter)

            if current_iter % opt['logger']['print_freq'] == 0:
                log_vars = {'epoch': epoch, 'iter': current_iter}
                log_vars.update({'lrs': model.get_current_learning_rate()})
                log_vars.update({
                    'time': iter_timer.get_avg_time(),
                    'data_time': data_timer.get_avg_time()})
                log_vars.update(model.get_current_log())
                msg_logger(log_vars)

            if current_iter % opt['logger']['save_checkpoint_freq'] == 0:
                logger.info('Saving models and training states.')
                model.save(epoch, current_iter)

            if (opt.get('val') is not None
                    and current_iter % opt['val']['val_freq'] == 0):
                for val_loader in val_loaders:
                    model.validation(
                        val_loader, current_iter, tb_logger,
                        opt['val']['save_img'])

            data_timer.start()
            iter_timer.start()
            train_data = prefetcher.next()

    consumed_time = str(
        datetime.timedelta(seconds=int(time.time() - start_time)))
    logger.info(f'End of training. Time consumed: {consumed_time}')
    logger.info('Save the latest model.')
    model.save(epoch=-1, current_iter=-1)
    if opt.get('val') is not None:
        for val_loader in val_loaders:
            model.validation(
                val_loader, current_iter, tb_logger,
                opt['val']['save_img'])
    if tb_logger:
        tb_logger.close()


if __name__ == '__main__':
    root_path = osp.abspath(osp.join(__file__, osp.pardir, osp.pardir))
    train_pipeline(root_path)
