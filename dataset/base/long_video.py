#!/usr/bin/env python3
# Copyright (C) Alibaba Group Holding Limited. 
""" Dataset for Untrimmed videos: untrimmed k400, hacs. """

import os, sys
import random
import torch
import torch.utils.data
import utils.logging as logging
import json
import time
import math
import numpy as np
import oss2 as oss
import pandas as pd
import torch.utils.dlpack as dlpack
from torchvision.transforms import Compose
import torchvision.transforms._transforms_video as transforms
import torch.nn.functional as F
from dataset.utils.transformations import (
    ColorJitter, KineticsResizedCrop
)
from tqdm import tqdm
from dataset.base.base_dataset import BaseVideoDataset

import utils.bucket as bu

from dataset.base.builder import DATASET_REGISTRY

logger = logging.get_logger(__name__)


def random_center(start, end):
    if end < start:
        end = start
    p = random.random()
    return (end - start) * p + start


@DATASET_REGISTRY.register()
class Longvideo(BaseVideoDataset):
    def __init__(self, cfg, split):
        self.construct_set = split
        self.split_table = {"train": "training",
                            "test": "testing",
                            "val": "validation"}
        self.video_name_prefix = 'v_'
        super(Longvideo, self).__init__(cfg, split)
        self.decode = self._decode_clip_video
        self._sampling_rate = self.cfg.DATA.SAMPLING_RATE
        if self.split == "test" and self.cfg.PRETRAIN.ENABLE == False:
            self._pre_transformation_config_required = True

    def _construct_dataset(self, cfg):
        subset_path = self.split_table[self.construct_set]
        self._construct_dataset_vid_clip_subset(cfg, subset_path)

    def _construct_dataset_vid_clip_subset(self, cfg, subset_path):
        if hasattr(self, "_samples") is False:
            self._samples = []
            self._video_clips_dict = {}
            self._video_name_clips = {}
        clips_list_file = '{}.txt'.format(subset_path)
        local_clips_file = os.path.join(cfg.OUTPUT_DIR, clips_list_file) + '{}'.format(cfg.LOCAL_RANK if hasattr(cfg, 'LOCAL_RANK') else 0)
        if 'k400' in self.data_root_dir:
            local_clips_file = self._get_object_to_file(os.path.join(self.anno_dir, clips_list_file), local_clips_file, read_from_buffer=False)
        else:
            local_clips_file = self._get_object_to_file(os.path.join(self.data_root_dir, clips_list_file), local_clips_file, read_from_buffer=False)
        videos_name_list = []
        videos_clip_cnt_list = []
        with open(local_clips_file, 'r') as f:
            for line in tqdm(f):
                line = line.strip()
                video_name, start_time, end_time = line.split(',')
                if len(videos_name_list) == 0 or videos_name_list[-1] != video_name:
                    videos_name_list.append(video_name)
                    videos_clip_cnt_list.append(0)
                videos_clip_cnt_list[-1] += 1
                full_name = '_'.join([video_name, start_time, end_time])
                if video_name not in self._video_name_clips:
                    self._video_name_clips[video_name] = []
                    self._samples.append(video_name)
                if 'k400' in self.data_root_dir:
                    self._video_clips_dict[(video_name, int(int(start_time)/1000))] = os.path.join(self.data_root_dir, self.video_name_prefix + full_name + '.mp4')
                    self._video_name_clips[video_name] += [os.path.join(self.data_root_dir, self.video_name_prefix + full_name + '.mp4')]
                else:
                    self._video_clips_dict[(video_name, int(int(start_time)/1000))] = os.path.join(self.data_root_dir, subset_path, self.video_name_prefix + full_name + '.mp4')
                    self._video_name_clips[video_name] += [os.path.join(self.data_root_dir, subset_path, self.video_name_prefix + full_name + '.mp4')]
        logger.info("Dataset {} split {} loaded. Length {}.".format(self.dataset_name, self.construct_set, len(self._samples)))

    def _get_sample_info(self, index):
        """
            Input: 
                index (int): video index
            Returns:
                sample_info (dict): contains different informations to be used later
                    Things that must be included are:
                    "video_path" indicating the video's path w.r.t. index
                    "supervised_label" indicating the class of the video 
        """
        video_name = self._samples[index]
        videos_list = self._video_name_clips[video_name]
        duration = float(videos_list[-1].split('_')[-1].split('.')[0])/1000
        label = {}
        if self.cfg.DATA.HICO_PLUS_PLUS.ENABLE:
            clips_for_video = self._get_hicopp_clips_time_stamp_dual(duration)
        else:
            clips_for_video = self._get_hico_clips_time_stamp(duration)
        sample_info = {
            "path": video_name,
            "clips_for_video": clips_for_video,
            "video_name": video_name,
            "supervised_label": label,
        }
        return sample_info

    def _pre_transformation_config(self):
        """
            Set transformation parameters if required.
        """
        self.resize_video.set_spatial_index(self.spatial_idx)

    def _custom_sampling(self, vid_length, clip_idx, num_clips, num_frames, interval=2, random_sample=True):
        return self._interval_based_sampling(vid_length, clip_idx, num_clips, num_frames, interval)

    def _get_sample_videos(self, video_name, sample_time):
        video_length = 5
        start_time, end_time = sample_time
        video_start_sec = math.floor(start_time / video_length) * video_length
        video_end_sec = math.ceil(end_time / video_length) * video_length
        videos_list = []
        for st in range(video_start_sec, video_end_sec, video_length):
            videos_list.append(self._video_clips_dict[(video_name, st)])
        return videos_list, [video_start_sec, video_end_sec]
    
    def _config_transform(self):
        self.transform = None
    
    def _decode_clips(self, clips, sample_info, index, num_clips_per_video):
        videos_path_list = []
        videos_se_list = []
        for sample_time in (clips):
            videos_path, [video_start_sec, video_end_sec] = self._get_sample_videos(sample_info['video_name'], sample_time)
            videos_path_list.append(videos_path)
            videos_se_list.append([video_start_sec, video_end_sec])

        vr_list = []
        file_to_remove_list = []
        for vps in videos_path_list:
            v_list = []
            for vp in vps:
                vr, file_to_remove, success =  self._read_video(vp, index)
                v_list.append(vr)
                file_to_remove_list.extend(file_to_remove)
                if not success:
                    for vr in v_list:
                        del vr
                    return None, file_to_remove_list, success
            vr_list.append(v_list)

        frame_list = []
        for idx in range(len(clips)):
            list_ = self._get_time_stamp_frames_list(
                sum([len(vr) for vr in vr_list[idx]]),
                vr_list[idx][0].get_avg_fps(),
                [clips[idx][0]-videos_se_list[idx][0], clips[idx][1]-videos_se_list[idx][0]]
            )
            frames_collect = []
            exist_len = 0
            for vr in vr_list[idx]:
                select_idx = (list_ >= exist_len) & (list_ < (exist_len + len(vr)))
                select_frames = list_[select_idx] - exist_len
                exist_len += len(vr)
                frames = dlpack.from_dlpack(vr.get_batch(select_frames).to_dlpack()).clone()
                frames_collect.append(frames)
            frame_list.append(torch.cat(frames_collect, dim=0))
        sec_list = [c[0] for c in clips]
        frames = torch.stack(frame_list)
        if num_clips_per_video == 1:
            frames = frames.squeeze(0)
        for idx in range(len(clips)):
            for vr in vr_list[idx]:
                del vr
        return {"video": frames, "sec": torch.tensor(sec_list).float()}, file_to_remove_list, True

    def _decode_clip_video(self, sample_info, index, num_clips_per_video=1):
        clips = sample_info['clips_for_video']
        return self._decode_clips(clips, sample_info, index, num_clips_per_video)

    def _get_dataset_list_name(self):
        return

    def _get_ssl_label(self, frames):
        return
