#!/usr/bin/env python3
# From https://github.com/pytorch/vision/blob/main/torchvision/transforms/transforms.py

# BSD 3-Clause License

# Copyright (c) Soumith Chintala 2016, 
# All rights reserved.

# Redistribution and use in source and binary forms, with or without
# modification, are permitted provided that the following conditions are met:

# * Redistributions of source code must retain the above copyright notice, this
#   list of conditions and the following disclaimer.

# * Redistributions in binary form must reproduce the above copyright notice,
#   this list of conditions and the following disclaimer in the documentation
#   and/or other materials provided with the distribution.

# * Neither the name of the copyright holder nor the names of its
#   contributors may be used to endorse or promote products derived from
#   this software without specific prior written permission.

# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
# AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
# IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
# DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE
# FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
# DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
# SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
# CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY,
# OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
# OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.


import torch
import math
import torchvision
import torchvision.transforms._functional_video as F
from torchvision.transforms import Lambda, Compose, RandomApply
import random
import numbers
from collections.abc import Sequence
from typing import Optional, Tuple, List

class RandomColorJitter(object):
    """Randomly change the brightness, contrast and saturation of an image.
    Args:
        brightness (float or tuple of float (min, max)): How much to jitter brightness.
            brightness_factor is chosen uniformly from [max(0, 1 - brightness), 1 + brightness]
            or the given [min, max]. Should be non negative numbers.
        contrast (float or tuple of float (min, max)): How much to jitter contrast.
            contrast_factor is chosen uniformly from [max(0, 1 - contrast), 1 + contrast]
            or the given [min, max]. Should be non negative numbers.
        saturation (float or tuple of float (min, max)): How much to jitter saturation.
            saturation_factor is chosen uniformly from [max(0, 1 - saturation), 1 + saturation]
            or the given [min, max]. Should be non negative numbers.
        hue (float or tuple of float (min, max)): How much to jitter hue.
            hue_factor is chosen uniformly from [-hue, hue] or the given [min, max].
            Should have 0<= hue <= 0.5 or -0.5 <= min <= max <= 0.5.
        grayscale
    """
    def __init__(
        self, brightness=0, contrast=0, saturation=0, hue=0, 
        grayscale=0, color=0, blur=0,
        consistent=False, shuffle=True, 
    ):
        self.brightness = self._check_input(brightness, 'brightness')
        self.contrast = self._check_input(contrast, 'contrast')
        self.saturation = self._check_input(saturation, 'saturation')
        self.hue = self._check_input(hue, 'hue', center=0, bound=(-0.5, 0.5),
                                     clip_first_on_zero=False)
        self.blur = blur
        self.color = color
        self.grayscale = grayscale

        self.consistent = consistent
        self.shuffle = shuffle

    def _check_input(self, value, name, center=1, bound=(0, float('inf')), clip_first_on_zero=True):
        if isinstance(value, numbers.Number):
            if value < 0:
                raise ValueError("If {} is a single number, it must be non negative.".format(name))
            value = [center - float(value), center + float(value)]
            if clip_first_on_zero:
                value[0] = max(value[0], 0.0)
        elif isinstance(value, (tuple, list)) and len(value) == 2:
            if not bound[0] <= value[0] <= value[1] <= bound[1]:
                raise ValueError("{} values should be between {}".format(name, bound))
        else:
            raise TypeError("{} should be a single number or a list/tuple with lenght 2.".format(name))

        # if value is 0 or (1., 1.) for brightness/contrast/saturation
        # or (0., 0.) for hue, do nothing
        if value[0] == value[1] == center:
            value = None
        return value

    def _get_transform(self, T, device):
        """Get a randomized transform to be applied on image.
        Arguments are same as that of __init__.
        Arg:
            T (int): number of frames. Used when consistent = False.
        Returns:
            Transform which randomly adjusts brightness, contrast and
            saturation in a random order.
        """
        transforms = []
        if random.uniform(0, 1) < self.color:
            if self.brightness is not None:
                if self.consistent:
                    brightness_factor = random.uniform(self.brightness[0], self.brightness[1])
                else:
                    brightness_factor = torch.empty([1, T, 1, 1], device=device).uniform_(self.brightness[0], self.brightness[1])
                transforms.append(Lambda(lambda frame: adjust_brightness(frame, brightness_factor)))
            
            if self.contrast is not None:
                if self.consistent:
                    contrast_factor = random.uniform(self.contrast[0], self.contrast[1])
                else:
                    contrast_factor = torch.empty([1, T, 1, 1], device=device).uniform_(self.contrast[0], self.contrast[1])
                transforms.append(Lambda(lambda frame: adjust_contrast(frame, contrast_factor)))
            
            if self.saturation is not None:
                if self.consistent:
                    saturation_factor = random.uniform(self.saturation[0], self.saturation[1])
                else:
                    saturation_factor = torch.empty([1, T, 1, 1], device=device).uniform_(self.saturation[0], self.saturation[1])
                transforms.append(Lambda(lambda frame: adjust_saturation(frame, saturation_factor)))
            
            if self.hue is not None:
                if self.consistent:
                    hue_factor = random.uniform(self.hue[0], self.hue[1])
                else:
                    hue_factor = torch.empty([T, 1, 1], device=device).uniform_(self.hue[0], self.hue[1])
                transforms.append(Lambda(lambda frame: adjust_hue(frame, hue_factor)))

            if self.shuffle:
                random.shuffle(transforms)
        
        if random.uniform(0, 1) < self.blur:
            transforms.append(GaussianBlur(kernel_size=1))
        if random.uniform(0, 1) < self.grayscale:
            transforms.append(Lambda(lambda frame: rgb_to_grayscale(frame)))
        
        transform = Compose(transforms)

        return transform

    def __call__(self, clip):
        """
        Args:
            clip (torch.tensor): Size is (C, T, H, W)
        Return:
            clip (torch.tensor): Size is (C, T, H, W)
        """
        
        raw_shape = clip.shape #(C, T, H, W)
        device = clip.device
        T = raw_shape[1]
        transform = self._get_transform(T, device)
        clip = transform(clip)
        assert clip.shape == raw_shape
        return clip #(C, T, H, W)

    def __repr__(self):
        format_string = self.__class__.__name__ + '('
        format_string += 'brightness={0}'.format(self.brightness)
        format_string += ', contrast={0}'.format(self.contrast)
        format_string += ', saturation={0}'.format(self.saturation)
        format_string += ', hue={0})'.format(self.hue)
        format_string += ', grayscale={0})'.format(self.grayscale)
        return format_string

class BatchRandomColorjitter(RandomColorJitter):
    def __init__(
        self, cfg):
        brightness=cfg.AUGMENTATION.BRIGHTNESS
        contrast=cfg.AUGMENTATION.CONTRAST
        saturation=cfg.AUGMENTATION.SATURATION
        hue=cfg.AUGMENTATION.HUE
        grayscale=cfg.AUGMENTATION.GRAYSCALE
        color=cfg.AUGMENTATION.COLOR
        consistent=cfg.AUGMENTATION.CONSISTENT
        shuffle=cfg.AUGMENTATION.SHUFFLE
        super(BatchRandomColorjitter, self).__init__(brightness, contrast, saturation, hue, grayscale, color, consistent, shuffle)
        self.mean = cfg.DATA.MEAN
        self.std = cfg.DATA.STD

    def __call__(self, clip):
        mean = torch.Tensor(self.mean)[None, :,None, None, None].to(clip.device)
        std = torch.Tensor(self.std)[None, :,None, None, None].to(clip.device)
        clip = clip * std + mean

        raw_shape = clip.shape #(b, C, T, H, W)
        device = clip.device
        T = raw_shape[2]
        transform = self._get_transform(T, device)
        transformed_list = []
        for b in range(raw_shape[0]):
            transformed_list.append(transform(clip[b]))
        new_clip = torch.stack(transformed_list, dim=0)
        new_clip = (new_clip - mean) / std
        assert new_clip.shape == raw_shape
        return new_clip #(C, T, H, W)


def _is_tensor_a_torch_image(input):
    return input.ndim >= 2

def _blend(img1, img2, ratio):
    # type: (Tensor, Tensor, float) -> Tensor
    bound = 1 if img1.dtype in [torch.half, torch.float32, torch.float64] else 255
    return (ratio * img1 + (1 - ratio) * img2).clamp(0, bound).to(img1.dtype)

def rgb_to_grayscale(img):
    # type: (Tensor) -> Tensor
    """Convert the given RGB Image Tensor to Grayscale.
    For RGB to Grayscale conversion, ITU-R 601-2 luma transform is performed which
    is L = R * 0.2989 + G * 0.5870 + B * 0.1140
    Args:
        img (Tensor): Image to be converted to Grayscale in the form [C, H, W].
    Returns:
        Tensor: Grayscale image.
        Args:
            clip (torch.tensor): Size is (T, H, W, C)
        Return:
            clip (torch.tensor): Size is (T, H, W, C)
    """
    orig_dtype = img.dtype
    rgb_convert = torch.tensor([0.299, 0.587, 0.114])
    
    assert img.shape[0] == 3, "First dimension need to be 3 Channels"
    if img.is_cuda:
        rgb_convert = rgb_convert.to(img.device)
    
    img = img.float().permute(1,2,3,0).matmul(rgb_convert).to(orig_dtype)
    return torch.stack([img, img, img], 0)

def _rgb2hsv(img):
    r, g, b = img.unbind(0)

    maxc, _ = torch.max(img, dim=0)
    minc, _ = torch.min(img, dim=0)
    
    eqc = maxc == minc
    cr = maxc - minc
    s = cr / torch.where(eqc, maxc.new_ones(()), maxc)
    cr_divisor = torch.where(eqc, maxc.new_ones(()), cr)
    rc = (maxc - r) / cr_divisor
    gc = (maxc - g) / cr_divisor
    bc = (maxc - b) / cr_divisor

    hr = (maxc == r) * (bc - gc)
    hg = ((maxc == g) & (maxc != r)) * (2.0 + rc - bc)
    hb = ((maxc != g) & (maxc != r)) * (4.0 + gc - rc)
    h = (hr + hg + hb)
    h = torch.fmod((h / 6.0 + 1.0), 1.0)
    return torch.stack((h, s, maxc))

def _hsv2rgb(img):
    l = len(img.shape)
    h, s, v = img.unbind(0)
    i = torch.floor(h * 6.0)
    f = (h * 6.0) - i
    i = i.to(dtype=torch.int32)
    
    p = torch.clamp((v * (1.0 - s)), 0.0, 1.0)
    q = torch.clamp((v * (1.0 - s * f)), 0.0, 1.0)
    t = torch.clamp((v * (1.0 - s * (1.0 - f))), 0.0, 1.0)
    i = i % 6

    if l == 3:
        tmp = torch.arange(6)[:, None, None]
    elif l == 4:
        tmp = torch.arange(6)[:, None, None, None]
    
    if img.is_cuda:
        tmp = tmp.to(img.device)

    mask = i == tmp #(H, W) == (6, H, W)

    a1 = torch.stack((v, q, p, p, t, v))
    a2 = torch.stack((t, v, v, q, p, p))
    a3 = torch.stack((p, p, t, v, v, q))
    a4 = torch.stack((a1, a2, a3)) #(3, 6, H, W)

    if l == 3:
        return torch.einsum("ijk, xijk -> xjk", mask.to(dtype=img.dtype), a4) #(C, H, W)
    elif l == 4:
        return torch.einsum("itjk, xitjk -> xtjk", mask.to(dtype=img.dtype), a4) #(C, T, H, W)

def adjust_brightness(img, brightness_factor):
    # type: (Tensor, float) -> Tensor
    if not _is_tensor_a_torch_image(img):
        raise TypeError('tensor is not a torch image.')

    return _blend(img, torch.zeros_like(img), brightness_factor)

def adjust_contrast(img, contrast_factor):
    # type: (Tensor, float) -> Tensor
    if not _is_tensor_a_torch_image(img):
        raise TypeError('tensor is not a torch image.')
    
    mean = torch.mean(rgb_to_grayscale(img).to(torch.float), dim=(-4, -2, -1), keepdim=True)

    return _blend(img, mean, contrast_factor)

def adjust_saturation(img, saturation_factor):
    # type: (Tensor, float) -> Tensor
    if not _is_tensor_a_torch_image(img):
        raise TypeError('tensor is not a torch image.')

    return _blend(img, rgb_to_grayscale(img), saturation_factor)

def adjust_hue(img, hue_factor):
    """Adjust hue of an image.
    The image hue is adjusted by converting the image to HSV and
    cyclically shifting the intensities in the hue channel (H).
    The image is then converted back to original image mode.
    `hue_factor` is the amount of shift in H channel and must be in the
    interval `[-0.5, 0.5]`.
    See `Hue`_ for more details.
    .. _Hue: https://en.wikipedia.org/wiki/Hue
    Args:
        img (Tensor): Image to be adjusted. Image type is either uint8 or float.
        hue_factor (float):  How much to shift the hue channel. Should be in
            [-0.5, 0.5]. 0.5 and -0.5 give complete reversal of hue channel in
            HSV space in positive and negative direction respectively.
            0 means no shift. Therefore, both -0.5 and 0.5 will give an image
            with complementary colors while 0 gives the original image.
    Returns:
         Tensor: Hue adjusted image.
    """
    if isinstance(hue_factor, float) and  not(-0.5 <= hue_factor <= 0.5):
        raise ValueError('hue_factor ({}) is not in [-0.5, 0.5].'.format(hue_factor))
    elif isinstance(hue_factor, torch.Tensor) and not ((-0.5 <= hue_factor).sum() == hue_factor.shape[0] and (hue_factor <= 0.5).sum() == hue_factor.shape[0]):
        raise ValueError('hue_factor ({}) is not in [-0.5, 0.5].'.format(hue_factor))

    if not _is_tensor_a_torch_image(img):
        raise TypeError('tensor is not a torch image.')

    orig_dtype = img.dtype
    if img.dtype == torch.uint8:
        img = img.to(dtype=torch.float32) / 255.0

    img = _rgb2hsv(img)
    h, s, v = img.unbind(0)
    h += hue_factor
    h = h % 1.0
    img = torch.stack((h, s, v))
    img_hue_adj = _hsv2rgb(img)

    if orig_dtype == torch.uint8:
        img_hue_adj = (img_hue_adj * 255.0).to(dtype=orig_dtype)

    return img_hue_adj

def _cast_squeeze_in(img: torch.Tensor, req_dtypes: List[torch.dtype]) -> Tuple[torch.Tensor, bool, bool, torch.dtype]:
    need_squeeze = False
    # make image NCHW
    if img.ndim < 4:
        img = img.unsqueeze(dim=0)
        need_squeeze = True

    out_dtype = img.dtype
    need_cast = False
    if out_dtype not in req_dtypes:
        need_cast = True
        req_dtype = req_dtypes[0]
        img = img.to(req_dtype)
    return img, need_cast, need_squeeze, out_dtype

def _cast_squeeze_out(img: torch.Tensor, need_cast: bool, need_squeeze: bool, out_dtype: torch.dtype):
    if need_squeeze:
        img = img.squeeze(dim=0)

    if need_cast:
        if out_dtype in (torch.uint8, torch.int8, torch.int16, torch.int32, torch.int64):
            # it is better to round before cast
            img = torch.round(img)
        img = img.to(out_dtype)

    return img

def _get_gaussian_kernel1d(kernel_size: int, sigma: float) -> torch.Tensor:
    ksize_half = (kernel_size - 1) * 0.5

    x = torch.linspace(-ksize_half, ksize_half, steps=kernel_size)
    pdf = torch.exp(-0.5 * (x / sigma).pow(2))
    kernel1d = pdf / pdf.sum()

    return kernel1d


def _get_gaussian_kernel2d(
        kernel_size: List[int], sigma: List[float], dtype: torch.dtype, device: torch.device
) -> torch.Tensor:
    kernel1d_x = _get_gaussian_kernel1d(kernel_size[0], sigma[0]).to(device, dtype=dtype)
    kernel1d_y = _get_gaussian_kernel1d(kernel_size[1], sigma[1]).to(device, dtype=dtype)
    kernel2d = torch.mm(kernel1d_y[:, None], kernel1d_x[None, :])
    return kernel2d


def gaussian_blur(img: torch.Tensor, kernel_size: List[int], sigma: List[float]) -> torch.Tensor:
    if not (isinstance(img, torch.Tensor)):
        raise TypeError('img should be Tensor. Got {}'.format(type(img)))

    dtype = img.dtype if torch.is_floating_point(img) else torch.float32
    kernel = _get_gaussian_kernel2d(kernel_size, sigma, dtype=dtype, device=img.device)
    kernel = kernel.expand(img.shape[-3], 1, kernel.shape[0], kernel.shape[1])

    img, need_cast, need_squeeze, out_dtype = _cast_squeeze_in(img, [kernel.dtype, ])

    # padding = (left, right, top, bottom)
    padding = [kernel_size[0] // 2, kernel_size[0] // 2, kernel_size[1] // 2, kernel_size[1] // 2]
    img = torch.nn.functional.pad(img, padding, mode="reflect")
    img = torch.nn.functional.conv2d(img, kernel, groups=img.shape[-3])
    img = _cast_squeeze_out(img, need_cast, need_squeeze, out_dtype)
    return img

class GaussianBlur(object):
    """Blurs image with randomly chosen Gaussian blur.
    The image can be a PIL Image or a Tensor, in which case it is expected
    to have [..., C, H, W] shape, where ... means an arbitrary number of leading
    dimensions

    Args:
        kernel_size (int or sequence): Size of the Gaussian kernel.
        sigma (float or tuple of float (min, max)): Standard deviation to be used for
            creating kernel to perform blurring. If float, sigma is fixed. If it is tuple
            of float (min, max), sigma is chosen uniformly at random to lie in the
            given range.

    Returns:
        PIL Image or Tensor: Gaussian blurred version of the input image.

    """

    def __init__(self, kernel_size, sigma=(0.1, 2.0)):
        super().__init__()
        self.kernel_size = _setup_size(kernel_size, "Kernel size should be a tuple/list of two integers")
        for ks in self.kernel_size:
            if ks <= 0 or ks % 2 == 0:
                raise ValueError("Kernel size value should be an odd and positive number.")

        if isinstance(sigma, numbers.Number):
            if sigma <= 0:
                raise ValueError("If sigma is a single number, it must be positive.")
            sigma = (sigma, sigma)
        elif isinstance(sigma, Sequence) and len(sigma) == 2:
            if not 0. < sigma[0] <= sigma[1]:
                raise ValueError("sigma values should be positive and of the form (min, max).")
        else:
            raise ValueError("sigma should be a single number or a list/tuple with length 2.")

        self.sigma = sigma

    @staticmethod
    def get_params(sigma_min: float, sigma_max: float) -> float:
        """Choose sigma for random gaussian blurring.

        Args:
            sigma_min (float): Minimum standard deviation that can be chosen for blurring kernel.
            sigma_max (float): Maximum standard deviation that can be chosen for blurring kernel.

        Returns:
            float: Standard deviation to be passed to calculate kernel for gaussian blurring.
        """
        return torch.empty(1).uniform_(sigma_min, sigma_max).item()


    def __call__(self, img):
        """
        Args:
            img (PIL Image or Tensor): image to be blurred.

        Returns:
            PIL Image or Tensor: Gaussian blurred image
        """
        sigma = self.get_params(self.sigma[0], self.sigma[1])
        img = img.transpose(0, 1)
        output = gaussian_blur(img, self.kernel_size, [sigma, sigma])
        return output.transpose(0, 1)


    def __repr__(self):
        s = '(kernel_size={}, '.format(self.kernel_size)
        s += 'sigma={})'.format(self.sigma)
        return self.__class__.__name__ + s

def _setup_size(size, error_msg):
    if isinstance(size, numbers.Number):
        return int(size), int(size)

    if isinstance(size, Sequence) and len(size) == 1:
        return size[0], size[0]

    if len(size) != 2:
        raise ValueError(error_msg)

    return size
