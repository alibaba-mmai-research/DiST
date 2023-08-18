#!/usr/bin/env python3
# Copyright (C) Alibaba Group Holding Limited. 

""" Kinetics400 dataset. """

import os
import random
import torch
import torch.utils.data
import utils.logging as logging
import torch.utils.dlpack as dlpack

import time
import oss2 as oss

from torchvision.transforms import Compose
import torchvision.transforms._transforms_video as transforms
import torch.nn.functional as F
from dataset.utils.transformations import (
    ColorJitter, 
    KineticsResizedCrop
)

from dataset.base.base_dataset import BaseVideoDataset

import utils.bucket as bu

from dataset.base.builder import DATASET_REGISTRY
from dataset.utils.random_erasing import RandomErasing

logger = logging.get_logger(__name__)

@DATASET_REGISTRY.register()
class Kinetics400(BaseVideoDataset):
    def __init__(self, cfg, split):
        super(Kinetics400, self).__init__(cfg, split) 
        if hasattr(self.cfg, 'HICO') and (self.cfg.HICO.GRAUDAL_SAMPLING.ENABLE or self.cfg.HICO.VCL.ENABLE or self.cfg.HICO.TCL.ENABLE):
            self.decode = self._decode_hico_clips_for_trimmed_videos
        if self.split == "test" and self.cfg.PRETRAIN.ENABLE == False:
            self._pre_transformation_config_required = True
        
    def _get_dataset_list_name(self):
        """
        Returns the list for the dataset. 
        Returns:
            name (str): name of the list to be read
        """
        name = "kinetics400_{}_list.txt".format(
            self.split,
        )
        logger.info("Reading video list from file: {}".format(name))
        return name

    def _get_sample_info(self, index):
        """
        Returns the sample info corresponding to the index.
        Args: 
            index (int): target index
        Returns:
            sample_info (dict): contains different informations to be used later
                "path": indicating the target's path w.r.t. index
                "supervised_label": indicating the class of the target 
        """
        video_path, class_, = self._samples[index].strip().split(" ")
        class_ = int(class_)
        video_path = os.path.join(self.data_root_dir, video_path)
        sample_info = {
            "path": video_path,
            "supervised_label": class_,
        }
        return sample_info
    
    def _config_transform(self):
        """
        Configs the transform for the dataset.
        For train, we apply random cropping, random horizontal flip, random color jitter (optionally),
            normalization and random erasing (optionally).
        For val and test, we apply controlled spatial cropping and normalization.
        The transformations are stored as a callable function to "self.transforms".
        
        Note: This is only used in the supervised setting.
            For self-supervised training, the augmentations are performed in the 
            corresponding generator.
        """
        self.transform = None
        if self.split == 'train' and not self.cfg.PRETRAIN.ENABLE:
            std_transform_list = [
                transforms.ToTensorVideo(),
                transforms.RandomHorizontalFlipVideo()
            ]
            if self.cfg.DATA.TRAIN_JITTER_SCALES[0] <= 1:
                std_transform_list += [transforms.RandomResizedCropVideo(
                        size=self.cfg.DATA.TRAIN_CROP_SIZE,
                        scale=[
                            self.cfg.DATA.TRAIN_JITTER_SCALES[0],
                            self.cfg.DATA.TRAIN_JITTER_SCALES[1]
                        ],
                        ratio=self.cfg.AUGMENTATION.RATIO
                    ),]
            else:
                std_transform_list += [KineticsResizedCrop(
                    short_side_range = [self.cfg.DATA.TRAIN_JITTER_SCALES[0], self.cfg.DATA.TRAIN_JITTER_SCALES[1]],
                    crop_size = self.cfg.DATA.TRAIN_CROP_SIZE,
                ),]
            # Add color aug
                        # Add color aug
            if self.cfg.AUGMENTATION.AUTOAUGMENT.ENABLE:
                from dataset.utils.auto_augment import creat_auto_augmentation
                if self.cfg.AUGMENTATION.AUTOAUGMENT.BEFORE_CROP:
                    std_transform_list.insert(-1, creat_auto_augmentation(self.cfg.AUGMENTATION.AUTOAUGMENT.TYPE, self.cfg.DATA.TRAIN_CROP_SIZE, self.cfg.DATA.MEAN))
                else:
                    std_transform_list.append(creat_auto_augmentation(self.cfg.AUGMENTATION.AUTOAUGMENT.TYPE, self.cfg.DATA.TRAIN_CROP_SIZE, self.cfg.DATA.MEAN))
            elif self.cfg.AUGMENTATION.COLOR_AUG:
                std_transform_list.append(
                    ColorJitter(
                        brightness=self.cfg.AUGMENTATION.BRIGHTNESS,
                        contrast=self.cfg.AUGMENTATION.CONTRAST,
                        saturation=self.cfg.AUGMENTATION.SATURATION,
                        hue=self.cfg.AUGMENTATION.HUE,
                        grayscale=self.cfg.AUGMENTATION.GRAYSCALE,
                        consistent=self.cfg.AUGMENTATION.CONSISTENT,
                        shuffle=self.cfg.AUGMENTATION.SHUFFLE,
                        gray_first=self.cfg.AUGMENTATION.GRAY_FIRST,
                        ),
                )
            std_transform_list += [
                transforms.NormalizeVideo(
                    mean=self.cfg.DATA.MEAN,
                    std=self.cfg.DATA.STD,
                    inplace=True
                ),
                RandomErasing(self.cfg)
            ]
            self.transform = Compose(std_transform_list)
        elif self.split == 'val' or self.split == 'test':
            self.resize_video = KineticsResizedCrop(
                    short_side_range = [self.cfg.DATA.TEST_SCALE, self.cfg.DATA.TEST_SCALE],
                    crop_size = self.cfg.DATA.TEST_CROP_SIZE,
                    num_spatial_crops = self.cfg.TEST.NUM_SPATIAL_CROPS
                )
            std_transform_list = [
                transforms.ToTensorVideo(),
                self.resize_video,
                transforms.NormalizeVideo(
                    mean=self.cfg.DATA.MEAN,
                    std=self.cfg.DATA.STD,
                    inplace=True
                )
            ]
            self.transform = Compose(std_transform_list)


    def _pre_transformation_config(self):
        """
        Set transformation parameters if required.
        """
        self.resize_video.set_spatial_index(self.spatial_idx)

    def _custom_sampling(self, vid_length, vid_fps, clip_idx, num_clips, num_frames, interval=2, random_sample=True):
        return self._interval_based_sampling(vid_length, vid_fps, clip_idx, num_clips, num_frames, interval)

    def _get_ssl_label(self, frames):
        return

    def _decode_hico_clips_for_trimmed_videos(self, sample_info, index, num_clips_per_video=1):
        """
        Decodes the video given the sample info.
        Args: 
            sample_info         (dict): containing the "path" key specifying the location of the video.
            index               (int):  for debug.
            num_clips_per_video (int):  number of clips to be decoded from each video. set to 2 for contrastive learning and 1 for others.
        Returns:
            data            (dict): key "video" for the video data.
            file_to_remove  (list): list of temporary files to be deleted or BytesIO objects to be closed.
            success         (bool): flag for the indication of success or not.
        """
        path = sample_info["path"]
        vr, file_to_remove, success =  self._read_video(path, index)

        if not success:
            return vr, file_to_remove, success

        duration = len(vr) / vr.get_avg_fps()
        clips_for_video = self._get_hico_clips_time_stamp(duration)
        frame_list= []
        assert len(clips_for_video) == num_clips_per_video
        for idx in range(num_clips_per_video):
            # for each clip in the video, 
            # a list is generated before decoding the specified frames from the video
            list_ = self._get_time_stamp_frames_list(
                len(vr),
                vr.get_avg_fps(),
                [clips_for_video[idx][0], clips_for_video[idx][1]]
            )
            frames = None
            frames = dlpack.from_dlpack(vr.get_batch(list_).to_dlpack()).clone()
            frame_list.append(frames)
        frames = torch.stack(frame_list)
        if num_clips_per_video == 1:
            frames = frames.squeeze(0)
        del vr
        return {"video": frames}, file_to_remove, True
