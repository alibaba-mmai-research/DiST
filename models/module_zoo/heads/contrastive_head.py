#!/usr/bin/env python3
# Copyright (C) Alibaba Group Holding Limited. 

""" Contrastive heads. """

import torch
import torch.nn as nn

from models.base.base_blocks import BaseHead
from models.base.base_blocks import HEAD_REGISTRY


@HEAD_REGISTRY.register()
class ContrastiveHead(BaseHead):
    def __init__(self, cfg):
        self.with_bn = cfg.PRETRAIN.CONTRASTIVE.HEAD_BN
        self.bn_mmt = cfg.BN.MOMENTUM
        super(ContrastiveHead, self).__init__(cfg)
    
    def _construct_head(
        self,
        dim,
        num_classes,
        dropout_rate,
        activation_func,
    ):
        self.global_avg_pool = nn.AdaptiveAvgPool3d(1)
        
        self.mlp = MLP(self.cfg)

    
    def forward(self, x, deep_x=None):
        out = {}
        logits = {}
        x = self.global_avg_pool(x)
        # (N, C, T, H, W) -> (N, T, H, W, C).
        x = x.permute((0, 2, 3, 4, 1))

        logits = self.mlp(x)
        
        return x, logits

class MLP(nn.Module):
    def __init__(
        self, 
        cfg, 
        dim_in_override=None, 
        dim_out_override=None, 
        normalize=True,
        nonlinear=False
    ):
        super(MLP, self).__init__()
        with_bn     = cfg.PRETRAIN.CONTRASTIVE.HEAD_BN
        final_bn    = cfg.PRETRAIN.CONTRASTIVE.FINAL_BN
        bn_mmt      = cfg.BN.MOMENTUM
        dim         = cfg.VIDEO.BACKBONE.NUM_OUT_FEATURES if dim_in_override is None else dim_in_override
        mid_dim     = cfg.PRETRAIN.CONTRASTIVE.HEAD_MID_DIM if dim_in_override is None else dim_in_override if dim_in_override<cfg.PRETRAIN.CONTRASTIVE.HEAD_MID_DIM else cfg.PRETRAIN.CONTRASTIVE.HEAD_MID_DIM
        out_dim     = cfg.PRETRAIN.CONTRASTIVE.HEAD_OUT_DIM if dim_out_override is None else dim_out_override
        self.normalize = normalize
    
        self.linear_a = nn.Linear(dim, mid_dim)
        if with_bn:
            self.linear_a_bn = nn.BatchNorm3d(mid_dim, eps=1e-3, momentum=bn_mmt)
        self.logits_a_relu = nn.ReLU(inplace=True)
        self.linear_b = nn.Linear(mid_dim, mid_dim)
        if with_bn:
            self.linear_b_bn = nn.BatchNorm3d(mid_dim, eps=1e-3, momentum=bn_mmt)

        self.logits_out_relu = nn.ReLU(inplace=True)

        self.logits_out_b2 = nn.Linear(mid_dim, out_dim)
        if final_bn or (nonlinear and with_bn):
            self.final_bn = nn.BatchNorm3d(out_dim, eps=1e-3, momentum=bn_mmt)
        
        if nonlinear:
            self.final_relu = nn.ReLU(inplace=True)
    
    def forward(self, x):
        if len(x.shape) == 2:
            x = x.reshape(x.shape[0], 1, 1, 1, x.shape[1])
        x = self.linear_a(x)
        if hasattr(self, 'linear_a_bn'):
            x = self.linear_a_bn(x.permute(0,4,1,2,3)).permute(0,2,3,4,1)
        x = self.logits_a_relu(x)
        x = self.linear_b(x)
        if hasattr(self, 'linear_b_bn'):
            x = self.linear_b_bn(x.permute(0,4,1,2,3)).permute(0,2,3,4,1)
        x = self.logits_out_relu(x)
        x = self.logits_out_b2(x)
        if hasattr(self, 'final_bn'):
            x = self.final_bn(x.permute(0,4,1,2,3)).permute(0,2,3,4,1)
        if hasattr(self, 'final_relu'):
            x = self.final_relu(x)
        x = x.view(x.shape[0], -1)
        if self.normalize: 
            x = torch.nn.functional.normalize(x, p=2, dim=-1)
        return x


@HEAD_REGISTRY.register()
class ContrastiveHeadTopicPred(BaseHead):
    def __init__(self, cfg):
        super(ContrastiveHeadTopicPred, self).__init__(cfg)
    
    def _construct_head(
        self,
        dim,
        num_classes,
        dropout_rate,
        activation_func,
    ):
        self.global_avg_pool = nn.AdaptiveAvgPool3d((1, 1, 1))
        self.mlp_vcl = MLP(self.cfg)
        self.mlp_tcl = MLPTopicalPredictorSingleGPU(self.cfg)

    def forward(self, x, deep_x=None):
        if len(x.shape) == 5:
            b, c, t = x.shape[:3]
            x = self.global_avg_pool(x)
            x = x.view(b, c)
        else:
            b, c = x.shape[:2]
        logits_vcl = self.mlp_vcl(x)
        logits_vcl = logits_vcl.view(b, -1)
        logits_tcl = self.mlp_tcl(x)
        return logits_tcl, logits_vcl


class MLPTopicalPredictorSingleGPU(BaseHead):
    def __init__(self, cfg):
        super(MLPTopicalPredictorSingleGPU, self).__init__(cfg)
    
    def _construct_head(
        self,
        dim,
        num_classes,
        dropout_rate,
        activation_func,
    ):
        self.global_avg_pool = nn.AdaptiveAvgPool3d(1)
        
        self.mlp = MLP(
            self.cfg
        )
        self.topical_predictor = nn.Sequential(nn.Linear(self.cfg.PRETRAIN.CONTRASTIVE.HEAD_OUT_DIM*2, 256),
                                           nn.ReLU(),
                                           nn.Linear(256, 1),)
    
    def forward(self, x):
        x = self.mlp(x)
        num_samples_per_gpu = x.shape[0]
        logits = x
        b, c = logits.size()
        topical_map1 = torch.cat([x[:, None, :].expand(num_samples_per_gpu, b, c), logits[None, :, :].expand(num_samples_per_gpu, b, c)], dim=-1)
        topical_map2 = torch.cat([x[None, :, :].expand(num_samples_per_gpu, b, c), logits[:, None, :].expand(num_samples_per_gpu, b, c)], dim=-1)
        logits = torch.cat([self.topical_predictor(topical_map1), self.topical_predictor(topical_map2)], dim=-1)
        return logits


@HEAD_REGISTRY.register()
class ContrastiveHeadTopicPredPlusPlus(BaseHead):
    def __init__(self, cfg):
        super(ContrastiveHeadTopicPredPlusPlus, self).__init__(cfg)
    
    def _construct_head(
        self,
        dim,
        num_classes,
        dropout_rate,
        activation_func,
    ):
        self.global_avg_pool = nn.AdaptiveAvgPool3d((1, 1, 1))
        self.mlp_vcl = MLP(self.cfg)
        self.mlp_tcl = MLPTopicalPredictorSingleGPUPlusPlus(self.cfg)

    def forward(self, x, deep_x=None):
        if len(x.shape) == 5:
            b, c, t = x.shape[:3]
            x = self.global_avg_pool(x)
            x = x.view(b, c)
        else:
            b, c = x.shape[:2]
        logits_vcl = self.mlp_vcl(x)
        logits_vcl = logits_vcl.view(b, -1)
        logits_tcl = self.mlp_tcl(x)
        return logits_tcl, logits_vcl


class MLPTopicalPredictorSingleGPUPlusPlus(BaseHead):
    def __init__(self, cfg):
        super(MLPTopicalPredictorSingleGPUPlusPlus, self).__init__(cfg)
    
    def _construct_head(
        self,
        dim,
        num_classes,
        dropout_rate,
        activation_func,
    ):
        self.global_avg_pool = nn.AdaptiveAvgPool3d(1)
        
        self.mlp = MLP(
            self.cfg
        )
        self.topical_predictor = nn.Sequential(nn.Linear(self.cfg.PRETRAIN.CONTRASTIVE.HEAD_OUT_DIM*2, 256),
                                           nn.ReLU(),
                                           nn.Linear(256, 1),)
    
    def forward(self, x):
        x = self.mlp(x)
        num_samples_per_gpu = x.shape[0]
        b, c = x.size()
        logits = x.view(b//2, 2, c).mean(dim=1)
        topical_map1 = torch.cat([logits[:, None, :].expand(num_samples_per_gpu//2, b//2, c), logits[None, :, :].expand(num_samples_per_gpu//2, b//2, c)], dim=-1)
        topical_map2 = torch.cat([logits[None, :, :].expand(num_samples_per_gpu//2, b//2, c), logits[:, None, :].expand(num_samples_per_gpu//2, b//2, c)], dim=-1)
        logits = torch.cat([self.topical_predictor(topical_map1), self.topical_predictor(topical_map2)], dim=-1)
        return logits

