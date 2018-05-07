import _init_paths

import time
import argparse
import logging
import pprint
import os
import sys
from config.config import config, update_config

def parse_args():
    parser = argparse.ArgumentParser(description='Train R-FCN network')
    # general
    parser.add_argument('--cfg', help='experiment configure file name', required=True, type=str)

    args, rest = parser.parse_known_args()
    # update config
    update_config(args.cfg)

    # training
    parser.add_argument('--frequent', help='frequency of logging', default=config.default.frequent, type=int)
    args = parser.parse_args()
    return args

args = parse_args()
curr_path = os.path.abspath(os.path.dirname(__file__))
sys.path.insert(0, os.path.join(curr_path, '../external/mxnet', config.MXNET_VERSION))

import cv2
import shutil
import numpy as np
import mxnet as mx

from function.train_rpn import train_rpn
from function.test_rpn import test_rpn
from function.train_rcnn import train_rcnn
from utils.combine_model import combine_model
from utils.create_logger import create_logger

def alternate_train(args, ctx, pretrained, epoch):
    # set up logger
    logger, output_path = create_logger(config.output_path, args.cfg, config.dataset.image_set)

    # basic config
    begin_epoch = 0

    logging.info('########## TRAIN RPN WITH IMAGENET INIT')
    rpn1_prefix = os.path.join(output_path, 'rpn1')

    if not os.path.exists(rpn1_prefix):
        os.makedirs(rpn1_prefix)

    logging.info('########## TRAIN rfcn WITH IMAGENET INIT AND RPN DETECTION')
    rfcn1_prefix = os.path.join(output_path, 'rfcn1')
    config.TRAIN.BATCH_IMAGES = config.TRAIN.ALTERNATE.RCNN_BATCH_IMAGES
    train_rcnn(config, config.dataset.dataset, config.dataset.image_set, config.dataset.root_path, config.dataset.dataset_path,
               args.frequent, config.default.kvstore, config.TRAIN.FLIP, config.TRAIN.SHUFFLE, config.TRAIN.RESUME,
               ctx, pretrained, epoch, rfcn1_prefix, begin_epoch, config.TRAIN.ALTERNATE.rfcn1_epoch, train_shared=False,
               lr=config.TRAIN.ALTERNATE.rfcn1_lr, lr_step=config.TRAIN.ALTERNATE.rfcn1_lr_step, proposal='rpn', logger=logger,
               output_path=rpn1_prefix)

def main():
    print('Called with argument:', args)
    ctx = [mx.gpu(int(i)) for i in config.gpus.split(',')]
    alternate_train(args, ctx, config.network.pretrained, config.network.pretrained_epoch)

if __name__ == '__main__':
    main()
