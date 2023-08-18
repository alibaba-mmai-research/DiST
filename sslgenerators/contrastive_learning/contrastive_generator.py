#!/usr/bin/env python3
# Copyright (C) Alibaba Group Holding Limited. 

import os
import torch
import random
import utils.logging as logging
import torchvision.transforms._functional_video as F

from torchvision.transforms import Compose
import torchvision.transforms._transforms_video as transforms
from sslgenerators.builder import SSL_GENERATOR_REGISTRY
from .augmentations import RandomColorJitter, GaussianBlur
logger = logging.get_logger(__name__)


@SSL_GENERATOR_REGISTRY.register()
class ContrastiveGenerator(object):
    """
    Generator for pseudo camera motions.
    """
    def __init__(self, cfg, split):
        self.cfg = cfg
        self.loss = cfg.PRETRAIN.LOSS
        self.crop_size = cfg.DATA.TRAIN_CROP_SIZE
        self.split = split

        if type(self.crop_size) is list:
            assert len(self.crop_size) <= 2
            if len(self.crop_size) == 2:
                assert self.crop_size[0] == self.crop_size[1]
                self.crop_size = self.crop_size[0]   

        self.config_transform()
        self.labels = {"contrastive": torch.tensor([i for i in range(cfg.PRETRAIN.NUM_CLIPS_PER_VIDEO)])}
        self.current_idx = 0

    def sample_generator(self, frames, index):
        out = []
        if len(frames['video'].size()) == 5:
            for i in range(frames['video'].size(0)):
                out.append(self.transform(frames['video'][i]))
        elif len(frames['video'].size()) == 4:
            out.append(self.transform(frames['video']))
            out.append(self.transform(frames['video']))
        else:
            raise NotImplementedError("unknown frames:{}".format(frames['video'].shape))
        frames['video'] = torch.stack(out)
        return frames

    def config_transform(self):
        std_transform_list = []
        if self.split == 'train' or self.split == 'val':
            # To tensor and normalize
            std_transform_list += [
                transforms.ToTensorVideo(),
            ]
            std_transform_list += [transforms.RandomResizedCropVideo(
                size=self.cfg.DATA.TRAIN_CROP_SIZE,
                scale=[
                    self.cfg.DATA.TRAIN_JITTER_SCALES[0]*self.cfg.DATA.TRAIN_JITTER_SCALES[0]/256.0/340.0,
                    self.cfg.DATA.TRAIN_JITTER_SCALES[1]*self.cfg.DATA.TRAIN_JITTER_SCALES[1]/256.0/340.0
                ],
                ratio=self.cfg.AUGMENTATION.RATIO)]
            # Add color aug
            std_transform_list.append(
                    RandomColorJitter(
                        brightness=self.cfg.AUGMENTATION.BRIGHTNESS,
                        contrast=self.cfg.AUGMENTATION.CONTRAST,
                        saturation=self.cfg.AUGMENTATION.SATURATION,
                        hue=self.cfg.AUGMENTATION.HUE,
                        grayscale=self.cfg.AUGMENTATION.GRAYSCALE,
                        color=self.cfg.AUGMENTATION.COLOR,
                        consistent=self.cfg.AUGMENTATION.CONSISTENT,
                        shuffle=self.cfg.AUGMENTATION.SHUFFLE,
                        blur=self.cfg.AUGMENTATION.BLUR,
                    )
            )
            std_transform_list += [
                transforms.NormalizeVideo(
                    mean=self.cfg.DATA.MEAN,
                    std=self.cfg.DATA.STD,
                    inplace=True
                ),
                transforms.RandomHorizontalFlipVideo(),
            ]
            self.transform = Compose(std_transform_list)
        elif self.split == 'test':
            std_transform_list += [
                transforms.ToTensorVideo(),
                transforms.NormalizeVideo(
                    mean=self.cfg.DATA.MEAN,
                    std=self.cfg.DATA.STD,
                    inplace=True
                )
            ]
            self.transform = Compose(std_transform_list)

    def __call__(self, frames, index):
        return self.sample_generator(frames, index), self.labels
