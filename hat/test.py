# flake8: noqa
"""
HAT test entry point.

Extends basicsr's test pipeline with automatic YAML config backup:
the option file used for the run is copied (with a timestamp header) into
experiments/<name>/ at the start of every test run, mirroring the behaviour
that hat/train.py already provides for training runs.
"""

import logging
import os.path as osp

import torch

import hat.archs
import hat.data
import hat.models

from basicsr.data import build_dataloader, build_dataset
from basicsr.models import build_model
from basicsr.utils import (get_env_info, get_root_logger, get_time_str,
                            make_exp_dirs)
from basicsr.utils.options import copy_opt_file, dict2str, parse_options


def test_pipeline(root_path):
    opt, args = parse_options(root_path, is_train=False)

    torch.backends.cudnn.benchmark = True

    # Create experiment / result directories
    make_exp_dirs(opt)

    # ── YAML backup ──────────────────────────────────────────────────────────
    # Copy the option file into experiments/<name>/ with a timestamp header
    # so every test run is fully reproducible from its output folder alone.
    copy_opt_file(args.opt, opt['path']['experiments_root'])

    # ── Logging ──────────────────────────────────────────────────────────────
    log_file = osp.join(
        opt['path']['log'],
        f"test_{opt['name']}_{get_time_str()}.log"
    )
    logger = get_root_logger(
        logger_name='basicsr', log_level=logging.INFO, log_file=log_file
    )
    logger.info(get_env_info())
    logger.info(dict2str(opt))

    # ── Build test dataloaders ───────────────────────────────────────────────
    test_loaders = []
    for _, dataset_opt in sorted(opt['datasets'].items()):
        test_set = build_dataset(dataset_opt)
        test_loader = build_dataloader(
            test_set, dataset_opt,
            num_gpu=opt['num_gpu'],
            dist=opt['dist'],
            sampler=None,
            seed=opt['manual_seed'],
        )
        logger.info(
            f"Number of test images in {dataset_opt['name']}: {len(test_set)}"
        )
        test_loaders.append(test_loader)

    # ── Build model and run validation ───────────────────────────────────────
    model = build_model(opt)
    for test_loader in test_loaders:
        test_set_name = test_loader.dataset.opt['name']
        logger.info(f'Testing {test_set_name}...')
        model.validation(
            test_loader,
            current_iter=opt['name'],
            tb_logger=None,
            save_img=opt['val']['save_img'],
        )


if __name__ == '__main__':
    root_path = osp.abspath(osp.join(__file__, osp.pardir, osp.pardir))
    test_pipeline(root_path)
