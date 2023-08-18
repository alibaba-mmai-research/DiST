#!/usr/bin/env python3
# Copyright (C) Alibaba Group Holding Limited. 

""" DiST Branch. """
import math
import torch
import torch.nn as nn
import numpy as np
from timm.models.layers import trunc_normal_
import torch.nn.functional as F
from collections import OrderedDict
from einops import rearrange, repeat
from models.base.clip import CrossAttentionBlockGenral, QuickGELU, LayerNorm


class IntegrationNetwork(nn.Module):
    def __init__(self, cfg, d_model):
        super().__init__()
        integration_dim = cfg.VIDEO.BACKBONE.DIST.INTEGRATION_DIM
        integration_dim = cfg.VIDEO.BACKBONE.DIST.INTEGRATION_DIM
        t_kernel_size = cfg.VIDEO.BACKBONE.DIST.TEMPORAL_KERNEL_SIZE
        mlp_ratio = cfg.VIDEO.BACKBONE.DIST.INTEGRATION_MLP_RATIO
        self.ffn = nn.Sequential(OrderedDict([
            ("c_fc", nn.Linear(integration_dim, int(integration_dim * mlp_ratio))),
            ("gelu", QuickGELU()),
            ("c_proj", nn.Linear(int(integration_dim * mlp_ratio), integration_dim))
        ]))
        spatial_temporal_mlp_ratio = cfg.VIDEO.BACKBONE.DIST.INTEGRATION_TEMPORAL_MLP_RATIO
        self.temporal_ffn = nn.Sequential(OrderedDict([
            ("c_fc1", nn.Conv3d(integration_dim, int(integration_dim * spatial_temporal_mlp_ratio), kernel_size=(1, 1, 1), padding=(0, 0, 0))),
            ("c_fc2", nn.Conv3d(int(integration_dim * spatial_temporal_mlp_ratio), int(integration_dim * spatial_temporal_mlp_ratio), kernel_size=(t_kernel_size, 1, 1), padding=(t_kernel_size//2, 0, 0))),
            ("gelu1", QuickGELU()),
            ("c_proj", nn.Conv3d(int(integration_dim * spatial_temporal_mlp_ratio), integration_dim, kernel_size=(1), padding=(0), stride=1))
        ]))
        self.ln = LayerNorm(integration_dim)
        self.ln_temporal = LayerNorm(integration_dim)
        self.num_frames = cfg.DATA.NUM_INPUT_FRAMES
        self.alpha = int(cfg.DATA.SPARSE_SAMPLE_ALPHA)

    def forward(self, x):
        l, bt, c = x.size()
        b, t = bt // (self.num_frames // self.alpha), self.num_frames // self.alpha
        t_x = self.ln_temporal(x).view(l, b, t, c).permute(1, 3, 2, 0).reshape(b, c, t, l, 1)
        t_x = self.temporal_ffn(t_x).flatten(3).permute(3, 0, 2, 1).flatten(1, 2)
        return self.ffn(self.ln(x)) + t_x


class TemporalNet(nn.Module):
    def __init__(self, cfg, d_model):
        super().__init__()
        temporal_dim = cfg.VIDEO.BACKBONE.DIST.TEMPORAL_DIM
        t_kernel_size = cfg.VIDEO.BACKBONE.DIST.TEMPORAL_KERNEL_SIZE
        mlp_ratio = cfg.VIDEO.BACKBONE.DIST.TEMPORAL_CONV_MLP_RATIO
        self.temporal_net = nn.Sequential(OrderedDict([
            ("c_fc1", nn.Conv3d(temporal_dim, int(temporal_dim * mlp_ratio), kernel_size=(t_kernel_size, 1, 1), padding=(t_kernel_size//2, 0, 0))),
            ("gelu1", QuickGELU()),
            ("c_fc2", nn.Conv3d(int(temporal_dim * mlp_ratio), temporal_dim, kernel_size=(1, 3, 3), padding=(0, 1, 1))),
        ]))
        self.gelu = QuickGELU()
        self.ln = LayerNorm(temporal_dim)
        self.num_frames = cfg.DATA.NUM_INPUT_FRAMES

    def forward(self, x):
        assert hasattr(self, 'ln')
        return self.gelu(x + self.temporal_net(self.ln(x.permute(0, 2, 3, 4, 1)).permute(0, 4, 1, 2, 3)))


class Temporal2IntegrationNetwork(nn.Module):
    def __init__(self, cfg, d_model):
        super().__init__()
        temporal_dim = cfg.VIDEO.BACKBONE.DIST.TEMPORAL_DIM
        integration_dim = cfg.VIDEO.BACKBONE.DIST.INTEGRATION_DIM
        self.alpha = int(cfg.DATA.SPARSE_SAMPLE_ALPHA)
        self.num_frames = cfg.DATA.NUM_INPUT_FRAMES
        self.linear_fuse = nn.Conv3d(temporal_dim, integration_dim, kernel_size=(self.alpha, 1, 1), padding=(0, 0, 0), stride=(self.alpha, 1, 1))
        self.cls_token = nn.Parameter(torch.zeros((1, 1, self.num_frames // self.alpha, integration_dim)))
        from timm.models.layers import trunc_normal_
        trunc_normal_(self.cls_token, std=0.02)

    def forward(self, x):
        x = self.linear_fuse(x).flatten(3)
        b, c, t, hw = x.size()
        x = x.permute(3, 0, 2, 1) # hw, b, t, c
        x = torch.cat([self.cls_token.repeat(1, b, 1, 1), x], dim=0)
        x = x.flatten(1, 2)
        return x



class Integration2TemporalNetwork(nn.Module):
    def __init__(self, cfg, d_model):
        super().__init__()
        integration_dim = cfg.VIDEO.BACKBONE.DIST.INTEGRATION_DIM
        temporal_dim = cfg.VIDEO.BACKBONE.DIST.TEMPORAL_DIM
        self.alpha = int(cfg.DATA.SPARSE_SAMPLE_ALPHA)
        self.num_frames = cfg.DATA.NUM_INPUT_FRAMES
        self.linear_fuse = nn.Linear(integration_dim, temporal_dim)

    def forward(self, x):
        x = self.linear_fuse(x[1:, ...])
        l, bt, c = x.size()
        b, t = bt // (self.num_frames // self.alpha), self.num_frames // self.alpha
        x = x.view(l, b, t, c).permute(1, 3, 2, 0).reshape(b, c, t, int(math.sqrt(l)), int(math.sqrt(l)))
        # return x.repeat_interleave(self.alpha // self.temporal_alpha, dim=2)
        return torch.nn.functional.upsample_nearest(x, size=(x.size(2)*(self.alpha), x.size(3), x.size(4)))


class SpatialTemporalAdaPoolingNetwork(nn.Module):
    def __init__(self, cfg, d_model, layer_id):
        super().__init__()
        mlp_ratio = 4
        self.num_frames = cfg.DATA.NUM_INPUT_FRAMES
        if hasattr(cfg.DATA, 'SPARSE_SAMPLE_ALPHA'):
            self.sparse_sample_alpha = cfg.DATA.SPARSE_SAMPLE_ALPHA
        else:
            self.sparse_sample_alpha = 1
        integration_dim = cfg.VIDEO.BACKBONE.DIST.INTEGRATION_DIM
        self.integration_dim = integration_dim
        self.temporal_transformer = CrossAttentionBlockGenral(integration_dim, integration_dim // 64)
        self.positional_embedding = nn.Parameter(torch.empty(1, self.num_frames // self.sparse_sample_alpha, integration_dim))
        from timm.models.layers import trunc_normal_
        trunc_normal_(self.positional_embedding, std=0.02)
        self.output_map_cls_token = nn.Sequential(OrderedDict([
            ("c_fc", nn.Linear(integration_dim, int(integration_dim * mlp_ratio))),
            ("gelu", QuickGELU()),
            ("c_proj", nn.Linear(int(integration_dim * mlp_ratio), integration_dim))
        ]))
        self.ln_out_temp_cls_token = LayerNorm(integration_dim)

        self.spatial_transformer = CrossAttentionBlockGenral(integration_dim, integration_dim // 64)
        self.output_map_spatial_cls_token = nn.Sequential(OrderedDict([
            ("c_fc", nn.Linear(integration_dim, int(integration_dim * mlp_ratio))),
            ("gelu", QuickGELU()),
            ("c_proj", nn.Linear(int(integration_dim * mlp_ratio), integration_dim))
        ]))
        self.ln_out_spat_cls_token = LayerNorm(integration_dim)


    def forward(self, prev_feat, top_cls_token, spatial_cls_token):
        l, bt, c = prev_feat.size()
        b, t = bt // (self.num_frames // self.sparse_sample_alpha), (self.num_frames // self.sparse_sample_alpha)

        if hasattr(self, 'spatial_transformer'):
            aggregated_spatial_cls_token = self.spatial_transformer(spatial_cls_token, prev_feat, prev_feat)
            spatial_cls_token = spatial_cls_token + aggregated_spatial_cls_token
            spatial_cls_token = spatial_cls_token + self.output_map_spatial_cls_token(self.ln_out_spat_cls_token(spatial_cls_token))
            cls_token = spatial_cls_token[0, ...].reshape(b, t, spatial_cls_token.size(-1))
            # aggregated_spatial_cls_token = self.output_map_spatial_cls_token(self.ln_out_spat_cls_token(aggregated_spatial_cls_token))
        else:
            cls_token = spatial_cls_token[0, ...].reshape(b, t, spatial_cls_token.size(-1))
            aggregated_spatial_cls_token = 0.0
    
        if hasattr(self, 'temporal_transformer'):
            if hasattr(self, 'positional_embedding'):
                cls_token = (cls_token + self.positional_embedding.to(prev_feat.dtype)).permute(1, 0, 2)
            else:
                cls_token = cls_token.permute(1, 0, 2)
            aggregated_cls_token = self.temporal_transformer(top_cls_token, cls_token, cls_token)
            top_cls_token = top_cls_token + aggregated_cls_token
            top_cls_token = top_cls_token + self.output_map_cls_token(self.ln_out_temp_cls_token(top_cls_token))

        return top_cls_token, spatial_cls_token


class DiSTNetwork(nn.Module):
    # DiSTv5_18_14
    def __init__(self, cfg, d_model, width, output_dim):
        super().__init__()
        self.cfg = cfg
        self.selected_layers = cfg.VIDEO.BACKBONE.DIST.SELECTED_LAYERS
        num_layers = len(self.selected_layers)
        temporal_dim = cfg.VIDEO.BACKBONE.DIST.TEMPORAL_DIM
        integration_dim = cfg.VIDEO.BACKBONE.DIST.INTEGRATION_DIM
        spatial_patch_size = cfg.VIDEO.BACKBONE.DIST.S_PATCH_SIZE
        temporal_patch_size = cfg.VIDEO.BACKBONE.DIST.T_PATCH_SIZE
        self.alpha = int(cfg.DATA.SPARSE_SAMPLE_ALPHA)

        self.temporal_stem = nn.Conv3d(3, temporal_dim, 
                                       kernel_size=(temporal_patch_size, spatial_patch_size, spatial_patch_size),
                                       stride=(1, spatial_patch_size, spatial_patch_size),
                                       padding=(temporal_patch_size//2, 0, 0))

        self.input_linears = nn.ModuleList([nn.Linear(d_model, integration_dim) for i in range(num_layers)])
        self.integration2temporal_nets = nn.ModuleList([Integration2TemporalNetwork(cfg, d_model=d_model) for i in range(num_layers)])
        self.temporal2integration_nets = nn.ModuleList([Temporal2IntegrationNetwork(cfg, d_model=d_model) for i in range(num_layers)])
        self.num_frames = cfg.DATA.NUM_INPUT_FRAMES
        self.temporal_nets = nn.ModuleList([TemporalNet(cfg, d_model=d_model) for i in self.selected_layers])
        self.integration_nets = nn.ModuleList([IntegrationNetwork(cfg, d_model=d_model) for i in self.selected_layers])

        num_temporal_layers = cfg.VIDEO.BACKBONE.DIST.ADA_POOLING_LAYERS
        self.adapooling_nets = nn.ModuleList([SpatialTemporalAdaPoolingNetwork(cfg, d_model=d_model, layer_id=0) for i in range(num_temporal_layers)])

        scale = integration_dim ** -0.5
        self.proj_spatial_cls_token = nn.Linear(d_model, integration_dim)
        self.ln_post = LayerNorm(integration_dim)
        self.proj = nn.Parameter(scale * torch.randn(integration_dim, output_dim))
        self.aggregated_cls_token = nn.Parameter(torch.zeros((1, 1, integration_dim)))
        self.aggregated_spatial_cls_token = nn.Parameter(torch.zeros((1, 1, integration_dim)))
        from timm.models.layers import trunc_normal_
        trunc_normal_(self.aggregated_cls_token, std=0.02)
        trunc_normal_(self.aggregated_spatial_cls_token, std=0.02)
        self.init_weights()

    def init_weights(self):
        self.apply(self._init_weights)

    def _init_weights(self, m):
        if hasattr(m, 'skip_init') and m.skip_init:
            return
        from timm.models.layers import trunc_normal_
        init_type = 'trunc_normal_'
        if isinstance(m, nn.Conv3d) or isinstance(m, nn.Conv2d) or isinstance(m, nn.Conv1d) or isinstance(m, nn.Linear):
            if init_type == 'trunc_normal_':
                trunc_normal_(m.weight, std=.02)
            elif init_type == 'xavier_uniform_':
                nn.init.xavier_uniform_(m.weight)
            else:
                raise NotImplementedError
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)

    def forward(self, input):
        l, bt, c = input['mid_feat']['img'][self.selected_layers[0]].size()
        b, t = bt // (self.num_frames // self.alpha), self.num_frames // self.alpha
        x_temporal = self.temporal_stem(input['images'].view(b, self.num_frames, 3, input['images'].size(2), input['images'].size(3)).permute(0, 2, 1, 3, 4))
        res_feat = 0.0
        for idx, layer_id in enumerate(self.selected_layers[:]):
            x_temporal = self.temporal_nets[idx](x_temporal)
            mid_feat = self.input_linears[idx](input['mid_feat']['img'][layer_id]) + res_feat

            updated_x_temporal = self.integration2temporal_nets[idx](mid_feat) + x_temporal
            updated_mid_feat = mid_feat + self.temporal2integration_nets[idx](x_temporal)

            res_feat = self.integration_nets[idx](updated_mid_feat)
            x_temporal = updated_x_temporal

        aggregated_cls_token = self.aggregated_cls_token.repeat(1, b, 1)
        aggregated_spatial_cls_token = self.aggregated_spatial_cls_token.repeat(1, bt, 1)
        current_layer_feat = res_feat + updated_mid_feat
        for adapooling_net in self.adapooling_nets:
            aggregated_cls_token, aggregated_spatial_cls_token = adapooling_net(current_layer_feat, aggregated_cls_token, aggregated_spatial_cls_token)
        aggregated_cls_token = aggregated_cls_token.permute(1, 0, 2)  # LND -> NLD
        x_logits = self.ln_post(aggregated_cls_token[:, 0, :] + self.proj_spatial_cls_token(input['mid_feat']['img'][layer_id][:1].view(b, t, c).mean(dim=1)))

        if self.proj is not None:
            cls_x = x_logits @ self.proj
        return cls_x, input



