#!/usr/bin/env python3
# Copyright (C) Alibaba Group Holding Limited. 

""" BaseVideoDataset object to be extended for specific dataset. """

import os, json
import random
import torch
import torchvision
import torch.nn as nn
import torch.utils.data
import torch.utils.dlpack as dlpack
import utils.logging as logging

import re
import abc
import time
import random
import decord
import traceback
import numpy as np
from PIL import Image
from decord import VideoReader
from decord import cpu, gpu
decord.bridge.set_bridge('native')

from torchvision.transforms import Compose

from sslgenerators.builder import build_ssl_generator

import utils.bucket as bu

logger = logging.get_logger(__name__)

class BaseVideoDataset(torch.utils.data.Dataset):
    """
    The BaseVideoDataset object provides a base object for all the video/image/video-text dataset.
    Abstract methods are provided for completion in the specific dataset.
    Necessary methods for all datasets such as "_decode_video", "_decode_image", 
    "__getitem__" (with standard procedure for loading the data) as well as sampling methods 
    such as "_interval_based_sampling" and "_segment_based_sampling" are implemented. 
    The specific video datasets can be extended from this dataset according to different needs.
    """
    def __init__(self, cfg, split):
        """
        For initialization of the dataset, the global cfg and the split need to provided.
        Args:
            cfg     (Config): The global config object.
            split   (str): The split, e.g., "train", "val", "test"
        """
        self.cfg            = cfg
        self.split          = split
        self.data_root_dir  = cfg.DATA.DATA_ROOT_DIR
        self.anno_dir       = cfg.DATA.ANNO_DIR

        if self.split in ["train", "val"]:
            self.dataset_name = cfg.TRAIN.DATASET
            self._num_clips = 1
        elif self.split in ["test", "submission"]:
            self.dataset_name = cfg.TEST.DATASET
            self._num_clips = cfg.TEST.NUM_ENSEMBLE_VIEWS * cfg.TEST.NUM_SPATIAL_CROPS
        else:
            raise NotImplementedError("Split not supported")

        self._num_frames = cfg.DATA.NUM_INPUT_FRAMES
        self._sampling_rate = cfg.DATA.SAMPLING_RATE

        self.gpu_transform = cfg.AUGMENTATION.USE_GPU       # whether or not to perform the transform on GPU

        self.decode = self._decode_video                    # decode function, decode videos by default

        self.buckets = {}

        # if set to true, _pre_transformation_config will be called before every transformations
        # this is used in the testset, where cropping positions are set for the controlled crop
        self._pre_transformation_config_required = False    
        self._construct_dataset(cfg)
        self._config_transform()

        if cfg.DATA.DATASET_LABEL_TEXT.ENABLE:
            if cfg.DATA.DATASET_LABEL_TEXT.TEXT_TYPE == "labels_only":
                self._load_dataset_labels(cfg)

        # configures the pre-training
        if self.cfg.PRETRAIN.ENABLE:
            self.ssl_generator = build_ssl_generator(self.cfg, split)
            # NUM_CLIPS_PER_VIDEO specifies the number of clips decoded for each video
            # for contrastive learning, NUM_CLIPS_PER_VIDEO=2
            # for other ssl, NUM_CLIPS_PER_VIDEO=1
            self.num_clips_per_video = cfg.PRETRAIN.NUM_CLIPS_PER_VIDEO

    def _load_dataset_labels(self, cfg, assign_labels_path=None):
        logger.info("Loading dataset labels for {} ".format(self.dataset_name))
        local_file = os.path.join(cfg.OUTPUT_DIR, 'labels{}.json'.format(cfg.LOCAL_RANK if cfg.NUM_GPUS*cfg.NUM_SHARDS > 1 else 0))
        labels_file = 'labels.json'
        local_file = self._get_object_to_file(os.path.join(self.anno_dir, labels_file), local_file)
        from dataset.utils.simple_tokenizer import tokenize
        with open(local_file, "r") as f:
            lines = json.load(f)
        labels2text = {}
        prompt_prefix = cfg.DATA.DATASET_LABEL_TEXT.PROMPT_PREFIX.strip()
        for text, idx in lines.items():
            idx = int(idx)
            if len(prompt_prefix) > 0:
                labels2text[idx] = tokenize([prompt_prefix + ' ' + text.replace('"', '').strip()])
            else:
                labels2text[idx] = tokenize([text.replace('"', '').strip()])
        self.text_tokenizers = torch.cat([labels2text[i] for i in range(len(labels2text))], dim=0)

    @abc.abstractmethod
    def _get_dataset_list_name(self):
        """
        Returns the list for the dataset. 
        Returns:
            name (str): name of the list to be read
        """
        raise NotImplementedError

    @abc.abstractmethod
    def _get_sample_info(self, index):
        """
        Returns the sample info corresponding to the index.
        Args: 
            index (int): target index
        Returns:
            sample_info (dict): contains different informations to be used later
                Things that must be included are:
                "path" indicating the target's path w.r.t. index
                "supervised_label" indicating the class of the target 
        """
        raise NotImplementedError

    @abc.abstractmethod
    def _get_ssl_label(self, frames):
        """
        Uses cfg to obtain ssl label.
        Returns:
            ssl_label (dict): self-supervised labels
        """
        raise NotImplementedError

    @abc.abstractmethod
    def _config_transform(self):
        """
        Uses cfg to config transforms and assign the transforms to self.transform
        Note: This is only used in the supervised setting.
            For self-supervised training, the augmentations are performed in the 
            corresponding generator.
        """
        self.transform = Compose([])
        raise NotImplementedError

    @abc.abstractmethod
    def _pre_transformation_config(self):
        """
            Set transformation parameters if required.
        """
        raise NotImplementedError
    
    @abc.abstractmethod
    def _custom_sampling(self, vid_length, vid_fps, clip_idx, num_clips, num_frames, interval=2, random_sample=True):
        raise NotImplementedError

    def _get_video_frames_list(self, vid_length, vid_fps, clip_idx, random_sample=True):
        """
        Returns the list of frame indexes in the video for decoding. 
        Args:
            vid_length (int): video length
            clip_idx (int): clip index, -1 if random sampling (interval based sampling)
            num_clips (int): overall number of clips for clip_idx != -1 (interval based sampling) 
            num_frames (int): number of frames to sample 
            interval (int): the step size for interval based sampling (interval based sampling)
            random_sample (int): whether to randomly sample one frame from each segment (segment based sampling)
        Returns:
            frame_id_list (list): indicates which frames to sample from the video
        """
        if self.cfg.PRETRAIN.ENABLE and self.split == "train":
            return self._custom_sampling(vid_length, vid_fps, clip_idx, self.cfg.TEST.NUM_ENSEMBLE_VIEWS, self._num_frames, self._sampling_rate, random_sample)
        else:
            if self.cfg.DATA.SAMPLING_MODE == "interval_based":
                return self._interval_based_sampling(vid_length, vid_fps, clip_idx, self.cfg.TEST.NUM_ENSEMBLE_VIEWS, self._num_frames, self._sampling_rate)
            elif self.cfg.DATA.SAMPLING_MODE == "segment_based":
                return self._segment_based_sampling(vid_length, clip_idx, self.cfg.TEST.NUM_ENSEMBLE_VIEWS, self._num_frames, random_sample)
            else:
                raise NotImplementedError

    def _construct_dataset(self, cfg):
        """
        Constructs the dataset according to the global config object.
        Currently supports reading from csv, json and txt.
        Args:
            cfg (Config): The global config object.
        """
        self._samples = []
        self._spatial_temporal_index = []
        dataset_list_name = self._get_dataset_list_name()

        try:
            logger.info("Loading {} dataset list for split '{}'...".format(self.dataset_name, self.split))
            local_file = os.path.join(cfg.OUTPUT_DIR, dataset_list_name)
            local_file = self._get_object_to_file(os.path.join(self.anno_dir, dataset_list_name), local_file)
            if local_file[-4:] == ".csv":
                import pandas
                lines = pandas.read_csv(local_file)
                for line in lines.values.tolist():
                    for idx in range(self._num_clips):
                        self._samples.append(line)
                        self._spatial_temporal_index.append(idx)
            elif local_file[-4:] == "json":
                import json
                with open(local_file, "r") as f:
                    lines = json.load(f)
                for line in lines:
                    for idx in range(self._num_clips):
                        self._samples.append(line)
                        self._spatial_temporal_index.append(idx)
            else:
                with open(local_file) as f:
                    lines = f.readlines()
                    for line in lines:
                        for idx in range(self._num_clips):
                            self._samples.append(line.strip())
                            self._spatial_temporal_index.append(idx)
            logger.info("Dataset {} split {} loaded. Length {}.".format(self.dataset_name, self.split, len(self._samples)))
        except:
            raise ValueError("Data list {} not found.".format(os.path.join(self.anno_dir, dataset_list_name)))
        
        # validity check    
        assert len(self._samples) != 0, "Empty sample list {}".format(os.path.join(self.anno_dir, dataset_list_name))

    def _read_video(self, video_path, index):
        """
        Wrapper for downloading the video and generating the VideoReader object for reading the video. 
        Args: 
            video_path (str): video path to read the video from. Can in OSS form or in local hard drives.
            index      (int):  for debug.
        Returns:
            vr              (VideoReader):  VideoReader object wrapping the video.
            file_to_remove  (list):         list of temporary files to be deleted or BytesIO objects to be closed.
            success         (bool):         flag for the indication of success or not.
        """
        tmp_file = str(round(time.time() * 1000)) + video_path.split('/')[-1]  
        try:
            vr = None
            tmp_file = self._get_object_to_file(video_path, tmp_file, read_from_buffer=True, num_retries=1 if self.split == "train" else 20)
            vr = VideoReader(tmp_file)
            success = True
        except:
            success = False
        file_to_remove = [tmp_file] if video_path[:3] == "oss" else [None] # if not downloaded from oss, then no files need to be removed
        return vr, file_to_remove, success

    def _decode_video(self, sample_info, index, num_clips_per_video=1):
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

        if self.split == "train":
            clip_idx = -1
            self.spatial_idx = -1
        elif self.split == "val":
            clip_idx = -1
            self.spatial_idx = 0
        elif self.split == "test" or self.split == "submission":
            clip_idx = self._spatial_temporal_index[index] // self.cfg.TEST.NUM_SPATIAL_CROPS
            if self.cfg.TEST.NUM_SPATIAL_CROPS == 1:
                self.spatial_idx = 0
            else:
                self.spatial_idx = self._spatial_temporal_index[index] % self.cfg.TEST.NUM_SPATIAL_CROPS

        frame_list= []
        for idx in range(num_clips_per_video):
            # for each clip in the video, 
            # a list is generated before decoding the specified frames from the video
            list_ = self._get_video_frames_list(
                len(vr),
                vr.get_avg_fps(),
                clip_idx,
                random_sample=True if self.split=="train" else False 
            )
            frames = None
            frames = dlpack.from_dlpack(vr.get_batch(list_).to_dlpack()).clone()
            frame_list.append(frames)
        frames = torch.stack(frame_list)
        if num_clips_per_video == 1:
            frames = frames.squeeze(0)
        del vr
        return {"video": frames}, file_to_remove, True

    def _read_image(self, path, index):
        """
        Wrapper for downloading the image and generating the PIL.Image object for reading the image. 
        Args: 
            path    (str): image path to read the image from. Can in OSS form or in local hard drives.
            index   (int):  for debug.
        Returns:
            img             (PIL.Image):    image object for further processing.
            file_to_remove  (list):         list of temporary files to be deleted or BytesIO objects to be closed.
            success         (bool):         flag for the indication of success or not.
        """
        tmp_file = str(round(time.time() * 1000)) + path.split('/')[-1]  
        for tmp in range(10):
            try:
                img = None
                tmp_file = self._get_object_to_file(path, tmp_file, read_from_buffer=True)
                if isinstance(tmp_file, str):
                    with open(tmp_file, 'rb') as f:
                        img = Image.open(f).convert('RGB')
                else:
                    img = Image.open(tmp_file).convert('RGB')
                success = True
                break
            except:
                success = False
        file_to_remove = [tmp_file] if path[:3] == "oss" else [None]
        return img, file_to_remove, success

    def _decode_image(self, sample_info, index, num_clips_per_video=1):
        """
        Decodes the image given the sample info.
        Args: 
            sample_info         (dict): containing the "path" key specifying the location of the image.
            index               (int):  for debug.
            num_clips_per_video (int):  number of clips to be decoded from each video. set to 2 for contrastive learning and 1 for others.
                                        specifically in this function, num_clips_per_video does not matter since all things to be decoded is one image.
        Returns:
            data            (dict): key "video" for the image data.
                                    because this is a video database, the images will be in the shape of 1,H,W,C before further processing.
            file_to_remove  (list): list of temporary files to be deleted or BytesIO objects to be closed.
            success         (bool): flag for the indication of success or not.
        """
        path = sample_info["path"]
        img, tmp_file, success = self._read_image(path, index)

        if not success:
            return None, tmp_file, success

        frame = torch.ByteTensor(torch.ByteStorage.from_buffer(img.tobytes())).view(img.size[1], img.size[0], len(img.getbands()))
        frame = frame.unsqueeze(0) # 1, H, W, C
        return {"video":frame}, tmp_file, True

    def __getitem__(self, index):
        """
        Gets the specified data.
        Args:
            index (int): the index of the data in the self._samples list.
        Returns:
            frames (dict): {
                "video": (tensor), 
                "text_embedding" (optional): (tensor)
            }
            labels (dict): {
                "supervised": (tensor),
                "self-supervised" (optional): (...)
            }
        """
        sample_info = self._get_sample_info(index)

        # decode the data
        retries = 1 if self.split == "train" else 10
        for retry in range(retries):
            try:
                data, file_to_remove, success = self.decode(
                    sample_info, index, num_clips_per_video=self.num_clips_per_video if hasattr(self, 'num_clips_per_video') else 1
                )
                break
            except Exception as e:
                success = False
                traceback.print_exc()
                logger.warning("Error at decoding. {}/{}. Vid index: {}, Vid path: {}".format(
                    retry+1, retries, index, sample_info["path"]
                ))

        if not success:
            return self.__getitem__(index - 1) if index != 0 else self.__getitem__(index + 1)

        if self.gpu_transform:
            for k, v in data.items():
                data[k] = v.cuda(non_blocking=True)
        if self._pre_transformation_config_required:
            self._pre_transformation_config()
        
        labels = {}
        labels["supervised"] = sample_info["supervised_label"] if "supervised_label" in sample_info.keys() else {}
        if self.cfg.PRETRAIN.ENABLE:
            # generates the different augmented samples for pre-training
            try:
                data, labels["self-supervised"] = self.ssl_generator(data, index)
            except Exception as e:
                traceback.print_exc()
                print("Error at Vid index: {}, Vid path: {}, Vid shape: {}".format(
                    index, sample_info["path"], data["video"].shape
                ))
                return self.__getitem__(index - 1) if index != 0 else self.__getitem__(index + 1)
        else:
            # augment the samples for supervised training
            labels["self-supervised"] = {}
            if "flow" in data.keys() and "video" in data.keys():
                data = self.transform(data)
            elif "video" in data.keys():
                data["video"] = self.transform(data["video"]) # C, T, H, W = 3, 16, 240, 320, RGB
        
        if (self.split == "train" and \
            not self.cfg.PRETRAIN.ENABLE and \
            "ssv2" in self.cfg.TRAIN.DATASET and \
            self.cfg.AUGMENTATION.SSV2_FLIP):
            if random.random() < 0.5:
                data["video"] = torchvision.transforms._functional_video.hflip(data["video"])
                label_transforms = {
                    86: 87,
                    87: 86,
                    93: 94,
                    94: 93,
                    166: 167,
                    167: 166
                }
                if labels["supervised"] in label_transforms.keys():
                    labels["supervised"] = label_transforms[labels["supervised"]]
        
        # if the model is SlowFast, generate two sets of inputs with different framerates.
        if self.cfg.VIDEO.BACKBONE.META_ARCH == "Slowfast":
            slow_idx = torch.linspace(0, data["video"].shape[1], data["video"].shape[1]//self.cfg.VIDEO.BACKBONE.SLOWFAST.ALPHA+1).long()[:-1]
            fast_frames = data["video"].clone()
            slow_frames = data["video"][:,slow_idx,:,:].clone()
            data["video"] = [slow_frames, fast_frames]
        bu.clear_tmp_file(file_to_remove)

        return data, labels, index, {}
    
    def _get_object_to_file(self, obj_file: str, local_file, read_from_buffer=False, num_retries=10):
        """
        Wrapper for downloading the file object.
        Args:
            obj_file         (str):  the target file to be downloaded (if it starts by "oss").
            local_file       (str):  the local file to store the downloaded file.
            read_from_butter (bool): whether or not to directly download to the memory
            num_retries      (int):  number of retries.
        Returns:
            str or BytesIO depending on the read_from_buffer flag
            if read_from_buffer==True:
                returns BytesIO 
            else:
                returns str (indicating the location of the specified file)
        """
        if obj_file[:3] == "oss":
            bucket_name = obj_file.split('/')[2]
            if bucket_name not in self.buckets.keys():
                self.buckets[bucket_name] = self._initialize_oss(bucket_name)
            if read_from_buffer:
                local_file = bu.read_from_buffer(
                    self.buckets[bucket_name],
                    obj_file,
                    bucket_name,
                    num_retries
                )
            else:
                bu.read_from_bucket(
                    self.buckets[bucket_name],
                    obj_file,
                    local_file,
                    bucket_name,
                    num_retries
                )
            return local_file
        else:
            return obj_file
    
    def _initialize_oss(self, bucket_name):
        """
        Initializes the oss bucket.
        Currently supporting two OSS accounts.
        """
        if hasattr(self.cfg.OSS, "SECONDARY_DATA_OSS") and\
            self.cfg.OSS.SECONDARY_DATA_OSS.ENABLE and\
            bucket_name in self.cfg.OSS.SECONDARY_DATA_OSS.BUCKETS:
            return bu.initialize_bucket(
                self.cfg.OSS.SECONDARY_DATA_OSS.KEY, 
                self.cfg.OSS.SECONDARY_DATA_OSS.SECRET,
                self.cfg.OSS.SECONDARY_DATA_OSS.ENDPOINT,
                bucket_name
            )
        else:
            return bu.initialize_bucket(
                self.cfg.OSS.KEY, 
                self.cfg.OSS.SECRET,
                self.cfg.OSS.ENDPOINT,
                bucket_name
            )

    def __len__(self):
        """
        Returns the number of samples.
        """
        if hasattr(self.cfg.TRAIN, "NUM_SAMPLES") and self.split == 'train':
            return self.cfg.TRAIN.NUM_SAMPLES
        else:
            return len(self._samples)

    # -------------------------------------- Sampling Utils --------------------------------------
    def _interval_based_sampling(self, vid_length, vid_fps, clip_idx, num_clips, num_frames, interval):
        """
        Generates the frame index list using interval based sampling.
        Args:
            vid_length  (int): the length of the whole video (valid selection range).
            vid_fps     (int): the original video fps
            clip_idx    (int): -1 for random temporal sampling, and positive values for sampling specific clip from the video
            num_clips   (int): the total clips to be sampled from each video. 
                                combined with clip_idx, the sampled video is the "clip_idx-th" video from "num_clips" videos.
            num_frames  (int): number of frames in each sampled clips.
            interval    (int): the interval to sample each frame.
        Returns:
            index (tensor): the sampled frame indexes
        """
        if num_frames == 1:
            index = [random.randint(0, vid_length-1)]
        else:
            # transform FPS
            clip_length = num_frames * interval * vid_fps / self.cfg.DATA.TARGET_FPS

            max_idx = max(vid_length - clip_length, 0)
            if clip_idx == -1: # random sampling
                start_idx = random.uniform(0, max_idx)
            else:
                if num_clips == 1:
                    start_idx = max_idx / 2
                else:
                    start_idx = max_idx * clip_idx / num_clips
            if self.cfg.DATA.MINUS_INTERVAL:
                end_idx = start_idx + clip_length - interval
            else:
                end_idx = start_idx + clip_length - 1

            index = torch.linspace(start_idx, end_idx, num_frames)
            index = torch.clamp(index, 0, vid_length-1).long()

        return index
    
    def _segment_based_sampling(self, vid_length, clip_idx, num_clips, num_frames, random_sample):
        """
        Generates the frame index list using segment based sampling.
        Args:
            vid_length    (int):  the length of the whole video (valid selection range).
            clip_idx      (int):  -1 for random temporal sampling, and positive values for sampling specific clip from the video
            num_clips     (int):  the total clips to be sampled from each video. 
                                    combined with clip_idx, the sampled video is the "clip_idx-th" video from "num_clips" videos.
            num_frames    (int):  number of frames in each sampled clips.
            random_sample (bool): whether or not to randomly sample from each segment. True for train and False for test.
        Returns:
            index (tensor): the sampled frame indexes
        """
        index = torch.zeros(num_frames)
        index_range = torch.linspace(0, vid_length, num_frames+1)
        for idx in range(num_frames):
            if random_sample:
                index[idx] = random.uniform(index_range[idx], index_range[idx+1])
            else:
                if num_clips == 1:
                    index[idx] = (index_range[idx] + index_range[idx+1]) / 2
                else:
                    index[idx] = index_range[idx] + (index_range[idx+1] - index_range[idx]) * (clip_idx+1) / num_clips
        index = torch.round(torch.clamp(index, 0, vid_length-1)).long()

        return index

    def _get_time_stamp_frames_list(self, vid_length, vid_fps, temporal_anno):
        num_frames = self._num_frames
        interval = self._sampling_rate
        start_loc, end_loc = max(int(temporal_anno[0]*vid_fps), 0), min(int(temporal_anno[1]*vid_fps), vid_length)
        index = torch.linspace(start_loc, end_loc, num_frames)
        index = torch.clamp(index, min=start_loc, max=end_loc-1).long()
        return index

    def _get_hico_clips_time_stamp(self, duration):
        def random_center(start, end):
            if end < start:
                end = start
            p = random.random()
            return (end - start) * p + start

        clip_time = self._num_frames * self._sampling_rate / self.cfg.DATA.TARGET_FPS
        if self.cfg.HICO.GRAUDAL_SAMPLING.ENABLE:
            max_dis = self.cfg.HICO.GRAUDAL_SAMPLING.MAX_DIS
            current_rate = self._get_current_epoch()
            max_dis = max_dis[0] + (max_dis[1] - max_dis[0]) * max(min(current_rate, 1.0), 0.0)
        elif self.cfg.HICO.VCL.ENABLE:
            max_dis = self.cfg.HICO.VCL.MAX_DIS
        else:
            max_dis = -1
        c_list = [random_center(clip_time, duration-clip_time)]

        for j in range(self.cfg.PRETRAIN.NUM_CLIPS_PER_VIDEO-1):
            if max_dis < 0:
                c_list.append(random_center(clip_time, duration-clip_time))
            else:
                c_list.append(random_center(max(c_list[-1] - max_dis, clip_time), min(c_list[-1] + max_dis, duration-clip_time)))
        if self.cfg.HICO.TCL.ENABLE:
            c_list[-1] = random_center(clip_time, duration-clip_time)
        clips_for_video = [[max(c-clip_time/2, 0), min(c+clip_time/2, duration)] for c in c_list]
        return clips_for_video

    def _get_current_epoch(self):
        path = "./current_epoch/"
        current_epoch, total_epoch = os.listdir(path)[-1].split("-")
        epoch_rate = float(current_epoch) / float(total_epoch)
        return epoch_rate 

    def _get_hicopp_clips_time_stamp_dual(self, duration):
        def random_center(start, end):
            if end < start:
                end = start
            p = random.random()
            return (end - start) * p + start

        clip_time = self._num_frames * self._sampling_rate / self.cfg.DATA.TARGET_FPS
        current_rate = None
        if self.cfg.HICO.GRAUDAL_SAMPLING.ENABLE:
            max_dis = self.cfg.HICO.GRAUDAL_SAMPLING.MAX_DIS
            current_rate = self._get_current_epoch()
            max_dis = max_dis[0] + (max_dis[1] - max_dis[0]) * max(min(current_rate, 1.0), 0.0)
        elif self.cfg.HICO.VCL.ENABLE:
            max_dis = self.cfg.HICO.VCL.MAX_DIS
        else:
            max_dis = -1.0
        c_list = []
        tcl_max_dis = self.cfg.HICO.TCL.MAX_DIS if hasattr(self.cfg.HICO.TCL, "MAX_DIS") else -1.0
        assert self.cfg.PRETRAIN.NUM_CLIPS_PER_VIDEO % 2 == 0
        for j in range(int(self.cfg.PRETRAIN.NUM_CLIPS_PER_VIDEO // 2)):
            if max_dis < 0.0:
                c_list.append(random_center(clip_time, duration-clip_time))
                c_list.append(random_center(clip_time, duration-clip_time))
            else:
                if tcl_max_dis >= 0 and len(c_list) > 0:
                    c_list.append(random_center(max(c_list[-1] - tcl_max_dis, clip_time), min(c_list[-1] + tcl_max_dis, duration-clip_time)))
                else:
                    c_list.append(random_center(clip_time, duration-clip_time))
                c_list.append(random_center(max(c_list[-1] - max_dis, clip_time), min(c_list[-1] + max_dis, duration-clip_time)))
        clips_for_video = [[max(c-clip_time/2, 0), min(c+clip_time/2, duration)] for c in c_list]
        assert len(c_list) == self.cfg.PRETRAIN.NUM_CLIPS_PER_VIDEO
        return clips_for_video
