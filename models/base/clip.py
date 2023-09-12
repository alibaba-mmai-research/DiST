'''
MIT License

Copyright (c) 2021 OpenAI

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
This file is modified from https://github.com/openai/CLIP/blob/main/clip/clip.py.
'''



from collections import OrderedDict
from typing import Tuple, Union
import math
import numpy as np
import torch
import torch.nn.functional as F
from torch import nn
import utils.logging as logging
from utils.registry import Registry
from timm.models.layers import trunc_normal_, drop_path
from copy import deepcopy
TEMPORALNET_REGISTRY = Registry("TemporalNet")
ATTEN_BLOCK_REGISTRY = Registry("AttentionBlock")
logger = logging.get_logger(__name__)


class ModifiedResNet(nn.Module):
    """
    A ResNet class that is similar to torchvision's but contains the following changes:
    - There are now 3 "stem" convolutions as opposed to 1, with an average pool instead of a max pool.
    - Performs anti-aliasing strided convolutions, where an avgpool is prepended to convolutions with stride > 1
    - The final pooling layer is a QKV attention instead of an average pool
    """

    def __init__(self, cfg, layers, output_dim, heads, input_resolution=224, width=64):
        super().__init__()
        self.cfg = cfg
        self.output_dim = output_dim
        self.input_resolution = input_resolution

        # the 3-layer stem
        self.conv1 = nn.Conv2d(3, width // 2, kernel_size=3, stride=2, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(width // 2)
        self.relu1 = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(width // 2, width // 2, kernel_size=3, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(width // 2)
        self.relu2 = nn.ReLU(inplace=True)
        self.conv3 = nn.Conv2d(width // 2, width, kernel_size=3, padding=1, bias=False)
        self.bn3 = nn.BatchNorm2d(width)
        self.relu3 = nn.ReLU(inplace=True)
        self.avgpool = nn.AvgPool2d(2)

        # residual layers
        self._inplanes = width  # this is a *mutable* variable used during construction
        self.layer1 = self._make_layer(width, layers[0], cfg=cfg)
        self.layer2 = self._make_layer(width * 2, layers[1], stride=2, cfg=cfg)
        self.layer3 = self._make_layer(width * 4, layers[2], stride=2, cfg=cfg)
        self.layer4 = self._make_layer(width * 8, layers[3], stride=2, cfg=cfg)

        embed_dim = width * 32  # the ResNet feature dimension
        self.attnpool = AttentionPool2d(input_resolution // 32, embed_dim, heads, output_dim)

    def _make_layer(self, planes, blocks, stride=1, cfg=None):
        if hasattr(cfg.VIDEO.BACKBONE, "TADA2D") and cfg.VIDEO.BACKBONE.TADA2D.ENABLE:
            bottleneck_block = BottleneckTada
        else:
            bottleneck_block = Bottleneck
        layers = [Bottleneck(self._inplanes, planes, stride, cfg=cfg)]

        self._inplanes = planes * bottleneck_block.expansion
        for _ in range(1, blocks):
            layers.append(bottleneck_block(self._inplanes, planes, cfg=cfg))

        return nn.Sequential(*layers)

    def forward(self, x, others=None):
        def stem(x):
            x = self.relu1(self.bn1(self.conv1(x)))
            x = self.relu2(self.bn2(self.conv2(x)))
            x = self.relu3(self.bn3(self.conv3(x)))
            x = self.avgpool(x)
            return x

        x = x.type(self.conv1.weight.dtype)
        x = stem(x)
        x, _ = self.layer1((x, others))
        x, _ = self.layer2((x, others))
        x, _ = self.layer3((x, others))
        x, _ = self.layer4((x, others))
        avg_x = self.attnpool(x)

        return avg_x, x


class ResidualAttentionBlock(nn.Module):
    def __init__(self, d_model: int, n_head: int, attn_mask: torch.Tensor = None, cfg=None, layer_id=0):
        super().__init__()
        self.layer_id = layer_id
        self.attn = nn.MultiheadAttention(d_model, n_head)
        self.ln_1 = LayerNorm(d_model)
        self.mlp = nn.Sequential(OrderedDict([
            ("c_fc", nn.Linear(d_model, d_model * 4)),
            ("gelu", QuickGELU()),
            ("c_proj", nn.Linear(d_model * 4, d_model))
        ]))
        self.ln_2 = LayerNorm(d_model)
        self.attn_mask = attn_mask
        self.is_image_transformer = attn_mask is None

    def attention(self, x: torch.Tensor):
        self.attn_mask = self.attn_mask.to(dtype=x.dtype, device=x.device) if self.attn_mask is not None else None
        return self.attn(x, x, x, need_weights=False, attn_mask=self.attn_mask)[0]

    def forward(self, input):
        x, others = input
        x = x + self.attention(self.ln_1(x))
        x = x + self.mlp(self.ln_2(x))
        return (x, others)



class CrossAttentionBlockGenral(nn.Module):
    def __init__(self, d_model: int, n_head: int, attn_mask: torch.Tensor = None, cfg=None, layer_id=0):
        super().__init__()
        self.layer_id = layer_id
        self.attn = nn.MultiheadAttention(d_model, n_head)
        self.ln_1 = LayerNorm(d_model)

    def forward(self, query, key, value):
        return self.attn(self.ln_1(query), self.ln_1(key), self.ln_1(value), need_weights=False)[0]


@ATTEN_BLOCK_REGISTRY.register()
class ResidualAttentionBlockMid(nn.Module):
    def __init__(self, d_model: int, n_head: int, attn_mask: torch.Tensor = None, cfg=None, layer_id=0):
        super().__init__()
        self.layer_id = layer_id
        self.attn = nn.MultiheadAttention(d_model, n_head)
        self.ln_1 = LayerNorm(d_model)
        self.mlp = nn.Sequential(OrderedDict([
            ("c_fc", nn.Linear(d_model, d_model * 4)),
            ("gelu", QuickGELU()),
            ("c_proj", nn.Linear(d_model * 4, d_model))
        ]))
        self.ln_2 = LayerNorm(d_model)
        self.attn_mask = attn_mask
        self.is_image_transformer = attn_mask is None

    def attention(self, x: torch.Tensor):
        self.attn_mask = self.attn_mask.to(dtype=x.dtype, device=x.device) if self.attn_mask is not None else None
        return self.attn(x, x, x, need_weights=False, attn_mask=self.attn_mask)[0]

    def forward(self, input):
        x, others = input
        x = x + self.attention(self.ln_1(x))
        x = x + self.mlp(self.ln_2(x))
        if others is not None and "mid_feat" in others:
            key_name = "img"
            if key_name in others["mid_feat"]:
                others["mid_feat"][key_name][self.layer_id] = x.clone()
        return (x, others)


class LayerNorm(nn.LayerNorm):
    """Subclass torch's LayerNorm to handle fp16."""

    def forward(self, x: torch.Tensor):
        orig_type = x.dtype
        ret = super().forward(x.type(torch.float32))
        return ret.type(orig_type)


class LayerNorm3d(nn.LayerNorm):
    """Subclass torch's LayerNorm to handle fp16."""

    def forward(self, x: torch.Tensor):
        orig_type = x.dtype
        ret = super().forward(x.type(torch.float32).permute(0, 2, 3, 4, 1))
        return ret.type(orig_type).permute(0, 4, 1, 2, 3)


class QuickGELU(nn.Module):
    def forward(self, x: torch.Tensor):
        return x * torch.sigmoid(1.702 * x)


class Transformer(nn.Module):
    def __init__(self, width: int, layers: int, heads: int, attn_mask: torch.Tensor = None, cfg=None):
        super().__init__()
        self.width = width
        self.layers = layers
        if cfg is None:
            self.resblocks = nn.Sequential(*[ResidualAttentionBlock(width, heads, attn_mask, cfg=cfg, layer_id=i) for i in range(layers)])
        else:
            self.resblocks = nn.Sequential(*[ATTEN_BLOCK_REGISTRY.get(cfg.VIDEO.BACKBONE.ATTEN_BLOCK)(width, heads, attn_mask, cfg=cfg, layer_id=i) for i in range(layers)])

    def forward(self, x: torch.Tensor, others=None):
        return self.resblocks((x, others))


class VisionTransformer(nn.Module):
    def __init__(self, cfg, input_resolution: int, patch_size: int, width: int, layers: int, heads: int, output_dim: int):
        super().__init__()
        self.cfg = cfg
        self.input_resolution = input_resolution
        self.output_dim = output_dim
        self.num_frames = cfg.DATA.NUM_INPUT_FRAMES
        if hasattr(cfg.VIDEO.BACKBONE, 'TUBE_STEM') and cfg.VIDEO.BACKBONE.TUBE_STEM.ENABLE:
            self.use_tube_stem = True
            tube_size = cfg.VIDEO.BACKBONE.TUBE_STEM.TUBE_SIZE
            tube_stride = cfg.VIDEO.BACKBONE.TUBE_STEM.TUBE_STRIDE
            self.conv1 = nn.Conv3d(in_channels=3, out_channels=width, kernel_size=(tube_size, patch_size, patch_size), stride=(tube_size, patch_size, patch_size), bias=False)
        else:
            self.use_tube_stem = False
            self.conv1 = nn.Conv2d(in_channels=3, out_channels=width, kernel_size=patch_size, stride=patch_size, bias=False)
        if hasattr(cfg.DATA, 'SPARSE_SAMPLE_ALPHA'):
            self.sparse_sample_alpha = cfg.DATA.SPARSE_SAMPLE_ALPHA
        else:
            self.sparse_sample_alpha = 1

        scale = width ** -0.5
        self.class_embedding = nn.Parameter(scale * torch.randn(width))
        self.positional_embedding = nn.Parameter(scale * torch.randn((input_resolution // patch_size) ** 2 + 1, width))
        self.ln_pre = LayerNorm(width)

        self.transformer = Transformer(width, layers, heads, cfg=cfg)

        self.ln_post = LayerNorm(width)
        self.proj = nn.Parameter(scale * torch.randn(width, output_dim))
        self.init_weights()

    def init_weights(self):
        self.apply(self._init_weights)

    def _init_weights(self, m):
        if hasattr(m, 'skip_init') and m.skip_init:
            return
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

    def forward(self, x: torch.Tensor, others=None):
        if self.use_tube_stem:
            bt, c, h, w = x.size()
            b, t = bt // self.num_frames, self.num_frames
            x = x.view(b, t, c, h, w).permute(0, 2, 1, 3, 4)
            x = self.conv1(x)
            x = x.permute(0, 2, 1, 3, 4).flatten(0, 1)
        else:
            x = self.conv1(x)  # shape = [*, width, grid, grid]
        x = x.reshape(x.shape[0], x.shape[1], -1)  # shape = [*, width, grid ** 2]
        x = x.permute(0, 2, 1)  # shape = [*, grid ** 2, width]
        x = torch.cat([self.class_embedding.to(x.dtype) + torch.zeros(x.shape[0], 1, x.shape[-1], dtype=x.dtype, device=x.device), x], dim=1)  # shape = [*, grid ** 2 + 1, width]
        x = x + self.positional_embedding.to(x.dtype)
        x = self.ln_pre(x)

        x = x.permute(1, 0, 2)  # NLD -> LND
        if others is not None and 'mid_feat' in others:
            others['mid_feat']['img'][-2] = x.clone()
        if self.sparse_sample_alpha > 1:
            l, bt, c = x.size()
            b, t = bt // self.num_frames, self.num_frames
            x = x.view(l, b, t, c)[:, :, ::self.sparse_sample_alpha, :].flatten(1, 2)
        if others is not None and 'mid_feat' in others and 'img' in others['mid_feat']:
            others['mid_feat']['img'][-1] = x.clone()
        if others is not None:
            others['visual_ln_post'] = self.ln_post
            others['visual_proj'] = self.proj
        x, _ = self.transformer(x, others)
        x = x.permute(1, 0, 2)  # LND -> NLD

        x_logits = self.ln_post(x[:, 0, :])

        if others is not None and 'mid_feat' in others:
            others['mid_feat']['img']['x_logits'] = x_logits.clone()
        if self.proj is not None:
            cls_x = x_logits @ self.proj
        
        return cls_x, x_logits, x[:, 1:,:], others


class CLIP(nn.Module):
    def __init__(self,
                 cfg,
                 embed_dim: int,
                 # vision
                 image_resolution: int,
                 vision_layers: Union[Tuple[int, int, int, int], int],
                 vision_width: int,
                 vision_patch_size: int,
                 # text
                 context_length: int,
                 vocab_size: int,
                 transformer_width: int,
                 transformer_heads: int,
                 transformer_layers: int
                 ):
        super().__init__()
        self.cfg = cfg
        self.context_length = context_length
        self.num_frames = cfg.DATA.NUM_INPUT_FRAMES
        self.freeze_text = cfg.VIDEO.BACKBONE.FREEZE_TEXT
        self.freeze_visual = cfg.VIDEO.BACKBONE.FREEZE_VISUAL
        self.num_classes = cfg.VIDEO.HEAD.NUM_CLASSES
        self.record_vis_mid_feat = cfg.VIDEO.BACKBONE.RECORD_VIS_MID_FEAT
        self.zero_shot_test = cfg.TEST.ZEROSHOT.ENABLE if hasattr(cfg.TEST, 'ZEROSHOT') else False
        self.cache_text_feat = None
        self.meta_arch_name = cfg.VIDEO.BACKBONE.META_ARCH_NAME
        self.text_features = None
        self.text_logits = None
        if hasattr(cfg.DATA, 'SPARSE_SAMPLE_ALPHA'):
            self.slow_sparse_sample_alpha = cfg.DATA.SPARSE_SAMPLE_ALPHA
        else:
            self.slow_sparse_sample_alpha = 1
        if isinstance(vision_layers, (tuple, list)):
            vision_heads = vision_width * 32 // 64
            self.visual = ModifiedResNet(
                cfg,
                layers=vision_layers,
                output_dim=embed_dim,
                heads=vision_heads,
                input_resolution=image_resolution,
                width=vision_width
            )
        else:
            vision_heads = vision_width // 64
            self.visual = VisionTransformer(
                cfg,
                input_resolution=image_resolution,
                patch_size=vision_patch_size,
                width=vision_width,
                layers=vision_layers,
                heads=vision_heads,
                output_dim=embed_dim
            )
        from models.module_zoo.branches.dist import DiSTNetwork
        self.dist_net = DiSTNetwork(cfg, d_model=vision_width, width=vision_width, output_dim=embed_dim)
        self.transformer = Transformer(
            width=transformer_width,
            layers=transformer_layers,
            heads=transformer_heads,
            attn_mask=self.build_attention_mask()
        )
        self.vocab_size = vocab_size
        self.token_embedding = nn.Embedding(vocab_size, transformer_width)
        self.positional_embedding = nn.Parameter(torch.empty(self.context_length, transformer_width))
        self.ln_final = LayerNorm(transformer_width)

        self.text_projection = nn.Parameter(torch.empty(transformer_width, embed_dim))
        self.logit_scale = nn.Parameter(torch.ones([]) * np.log(1 / 0.07))

        self.initialize_parameters()

    def initialize_parameters(self):
        nn.init.normal_(self.token_embedding.weight, std=0.02)
        nn.init.normal_(self.positional_embedding, std=0.01)

        if isinstance(self.visual, ModifiedResNet):
            if self.visual.attnpool is not None:
                std = self.visual.attnpool.c_proj.in_features ** -0.5
                nn.init.normal_(self.visual.attnpool.q_proj.weight, std=std)
                nn.init.normal_(self.visual.attnpool.k_proj.weight, std=std)
                nn.init.normal_(self.visual.attnpool.v_proj.weight, std=std)
                nn.init.normal_(self.visual.attnpool.c_proj.weight, std=std)

            for resnet_block in [self.visual.layer1, self.visual.layer2, self.visual.layer3, self.visual.layer4]:
                for name, param in resnet_block.named_parameters():
                    if name.endswith("bn3.weight"):
                        nn.init.zeros_(param)

        proj_std = (self.transformer.width ** -0.5) * ((2 * self.transformer.layers) ** -0.5)
        attn_std = self.transformer.width ** -0.5
        fc_std = (2 * self.transformer.width) ** -0.5
        for block in self.transformer.resblocks:
            nn.init.normal_(block.attn.in_proj_weight, std=attn_std)
            nn.init.normal_(block.attn.out_proj.weight, std=proj_std)
            nn.init.normal_(block.mlp.c_fc.weight, std=fc_std)
            nn.init.normal_(block.mlp.c_proj.weight, std=proj_std)

        if self.text_projection is not None:
            nn.init.normal_(self.text_projection, std=self.transformer.width ** -0.5)

    def build_attention_mask(self):
        # lazily create causal attention mask, with full attention between the vision tokens
        # pytorch uses additive attention mask; fill with -inf
        mask = torch.empty(self.context_length, self.context_length)
        mask.fill_(float("-inf"))
        mask.triu_(1)  # zero out the lower diagonal
        return mask

    @property
    def dtype(self):
        return self.visual.conv1.weight.dtype

    def encode_image(self, image, others=None):
        return self.visual(image.type(self.dtype), others)

    def encode_text(self, text, others):
        x = self.token_embedding(text).type(self.dtype)  # [batch_size, n_ctx, d_model]
        x = x + self.positional_embedding.type(self.dtype)

        x = x.permute(1, 0, 2)  # NLD -> LND
        x, others = self.transformer(x, others)
        x = x.permute(1, 0, 2)  # LND -> NLD

        # x.shape = [batch_size, n_ctx, transformer.width]
        # take features from the eot embedding (eot_token is the highest number in each sequence)
        x_logits = x[torch.arange(x.shape[0]), text.argmax(dim=-1)]

        x = x_logits
        x = self.ln_final(x).type(self.dtype)
        x = x @ self.text_projection

        return x, x_logits, others

    def cache_text(self, text, others):
        if others is not None and 'label_embeddings' in others:
            return others['label_embeddings'], None, others
        else:
            if self.text_features is None or text.size(0) != self.text_features.size(0):
                self.transformer.eval()
                self.ln_final.eval()
                with torch.no_grad():
                    text_features, text_logits, others = self.encode_text(text, others)
                self.text_features, self.text_logits = text_features.clone(), text_logits.clone()
            elif isinstance(self.text_features, nn.Parameter):
                text_features = self.text_features.clone().detach()
                text_logits = None
            else:
                text_features, text_logits = self.text_features.clone(), self.text_logits.clone()
            return (text_features, text_logits, others)

    def cache_visual(self, image, others):
        self.visual.eval()
        with torch.no_grad():
            output = self.visual(image.type(self.dtype), others)
        return output

    def forward(self, image, text, others=None):
        if text is not None:
            return self.forward_with_text(image, text, others)
        else:
            return self.forward_without_text(image)

    def forward_without_text(self, image):
        others = {}
        if self.record_vis_mid_feat:
            if 'mid_feat' not in others:
                others['mid_feat'] = {}
            others['mid_feat']['img'] = {}
            others['attn_maps'] = {}
        if self.freeze_visual:
            img_cls_tokens, img_logits, img_features, others = self.cache_visual(image, others=others)
        else:
            img_cls_tokens, img_logits, img_features, others = self.encode_image(image, others=others)
        if hasattr(self, 'dist_net'):
            others['images'] = image
            img_cls_tokens, others = self.dist_net(others)
        return img_cls_tokens[:, None, :]

    def forward_with_text(self, image, text, others=None):
        if self.freeze_text:
            temporal_text_features, temporal_text_logits, others = self.cache_text(text, others)
        else:
            temporal_text_features, temporal_text_logits, others = self.encode_text(text, others)

        cls_text_features, cls_text_logits = temporal_text_features, temporal_text_logits

        if self.record_vis_mid_feat:
            if others is None:
                others = {}
            if 'mid_feat' not in others:
                others['mid_feat'] = {}
            others['mid_feat']['img'] = {}
            others['attn_maps'] = {}
            others['training'] = self.training
        if self.freeze_visual:
            img_cls_tokens, img_logits, img_features, others = self.cache_visual(image, others=others)
        else:
            img_cls_tokens, img_logits, img_features, others = self.encode_image(image, others=others)
        img_cls_tokens_ori = img_cls_tokens
        if hasattr(self, 'dist_net'):
            # temporal_text_features = torch.zeros((self.num_classes, 512), device=image.device)
            others['text_features'] = temporal_text_features
            others['images'] = image
            img_cls_tokens, others = self.dist_net(others)
            cls_text_features = others['text_features']
        if cls_text_features is not None:
            # normalized features
            img_cls_tokens = img_cls_tokens / img_cls_tokens.norm(dim=1, keepdim=True)
            cls_text_features = cls_text_features / cls_text_features.norm(dim=1, keepdim=True)
            cls_text_features = cls_text_features.type(img_cls_tokens.dtype)

            # cosine similarity as logits
            logit_scale = self.logit_scale.exp()
            logits_per_image = logit_scale * img_cls_tokens @ cls_text_features.t()
            logits_per_text = logits_per_image.t()
            if (self.zero_shot_test and not self.training) or self.prediction_fusion_enable:
                img_cls_tokens_ori = img_cls_tokens_ori / img_cls_tokens_ori.norm(dim=1, keepdim=True)
                logits_per_image_ori = logit_scale * img_cls_tokens_ori @ cls_text_features.t()
                logits_per_image_ori = logits_per_image_ori.reshape(logits_per_image.shape[0], -1, logits_per_image.shape[1]).mean(dim=1)
                if self.prediction_fusion_gating_enable:
                    w = (self.prediction_fusion_gating / self.prediction_fusion_gating_temp).sigmoid()
                else:
                    w = 0.5
                logits_per_image = logits_per_image * w + logits_per_image_ori * (1- w)
        else:
            logits_per_image, logits_per_text = None, None

            # shape = [global_batch_size, global_batch_size]
        output_dict = {"logits_per_image": logits_per_image, "logits_per_text": logits_per_text, "img_logits": img_cls_tokens_ori, "vid_logits": img_cls_tokens[:, None, :]}
        return output_dict

    def load_state_dict(self, state_dict, strict, first_init=False):
        mismatch = super().load_state_dict(state_dict, strict=strict)
        return mismatch


def convert_weights(model: nn.Module):
    """Convert applicable model parameters to fp16"""

    def _convert_weights_to_fp16(l):
        if isinstance(l, (nn.Conv1d, nn.Conv2d, nn.Conv3d, nn.Linear)):
            l.weight.data = l.weight.data.half()
            if l.bias is not None:
                l.bias.data = l.bias.data.half()

        if isinstance(l, nn.MultiheadAttention):
            for attr in [*[f"{s}_proj_weight" for s in ["in", "q", "k", "v"]], "in_proj_bias", "bias_k", "bias_v"]:
                tensor = getattr(l, attr)
                if tensor is not None:
                    tensor.data = tensor.data.half()

        for name in ["text_projection", "proj"]:
            if hasattr(l, name):
                attr = getattr(l, name)
                if attr is not None:
                    attr.data = attr.data.half()

    model.apply(_convert_weights_to_fp16)


def build_model(cfg, state_dict: dict):
    vit = "visual.proj" in state_dict

    if vit:
        vision_width = state_dict["visual.conv1.weight"].shape[0]
        vision_layers = len([k for k in state_dict.keys() if k.startswith("visual.") and k.endswith(".attn.in_proj_weight")])
        vision_patch_size = state_dict["visual.conv1.weight"].shape[-1]
        grid_size = round((state_dict["visual.positional_embedding"].shape[0] - 1) ** 0.5)
        image_resolution = vision_patch_size * grid_size
    else:
        counts: list = [len(set(k.split(".")[2] for k in state_dict if k.startswith(f"visual.layer{b}"))) for b in [1, 2, 3, 4]]
        vision_layers = tuple(counts)
        vision_width = state_dict["visual.layer1.0.conv1.weight"].shape[0]
        output_width = round((state_dict["visual.attnpool.positional_embedding"].shape[0] - 1) ** 0.5)
        vision_patch_size = None
        assert output_width ** 2 + 1 == state_dict["visual.attnpool.positional_embedding"].shape[0]
        image_resolution = output_width * 32

    if hasattr(cfg.VIDEO.BACKBONE, 'TUBE_STEM') and cfg.VIDEO.BACKBONE.TUBE_STEM.ENABLE:
        tube_size = cfg.VIDEO.BACKBONE.TUBE_STEM.TUBE_SIZE
        tube_stride = cfg.VIDEO.BACKBONE.TUBE_STEM.TUBE_STRIDE
        state_dict["visual.conv1.weight"] = state_dict["visual.conv1.weight"][:, :, None, :, :].repeat(1, 1, tube_size, 1, 1) / tube_size

    embed_dim = state_dict["text_projection"].shape[1]
    context_length = state_dict["positional_embedding"].shape[0]
    vocab_size = state_dict["token_embedding.weight"].shape[0]
    transformer_width = state_dict["ln_final.weight"].shape[0]
    transformer_heads = transformer_width // 64
    transformer_layers = len(set(k.split(".")[2] for k in state_dict if k.startswith("transformer.resblocks")))

    model = CLIP(
        cfg,
        embed_dim,
        image_resolution, vision_layers, vision_width, vision_patch_size,
        context_length, vocab_size, transformer_width, transformer_heads, transformer_layers
    )

    for key in ["input_resolution", "context_length", "vocab_size"]:
        if key in state_dict:
            del state_dict[key]
    if cfg.TRAIN.HALF_PRECISION:
        convert_weights(model)
    if hasattr(cfg.TRAIN, "CONVERT_LADDER_NET_WEIGHTS") and cfg.TRAIN.CONVERT_LADDER_NET_WEIGHTS:
        state_dict = convert_ladder_net_weights(state_dict)
    mismatch = model.load_state_dict(state_dict, strict=False, first_init=True)
    logger.info("Keys in model not matched: {}".format(mismatch[0]))
    logger.info("Keys in checkpoint not matched: {}".format(mismatch[1]))
    return model.eval()


def load(cfg):
    oss_model_path = cfg.VIDEO.BACKBONE.PRETRAIN_WEIGHT_PATH
    model_path = cfg.VIDEO.BACKBONE.PRETRAIN_WEIGHT_PATH
    if hasattr(cfg.VIDEO.BACKBONE, 'LOCAL_PRETRAIN_WEIGHT_PATH'):
        import os
        if os.path.exists(cfg.VIDEO.BACKBONE.LOCAL_PRETRAIN_WEIGHT_PATH):
            model_path = cfg.VIDEO.BACKBONE.LOCAL_PRETRAIN_WEIGHT_PATH
    if model_path.startswith("oss:"):
        from utils.checkpoint import download_model_from_bucket
        model_path = download_model_from_bucket(cfg, model_path)
    if oss_model_path.endswith('.pyth'):
        state_dict = torch.load(model_path, map_location="cpu")
    else:
        state_dict = torch.jit.load(model_path, map_location="cpu").state_dict()
    clip_model = build_model(cfg, state_dict)
    return clip_model


