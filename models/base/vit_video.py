from functools import partial
import numpy as np
import torch, math
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange, repeat
from timm.models.layers import drop_path, to_2tuple, trunc_normal_
from timm.models.registry import register_model
from timm.models.layers import trunc_normal_ as __call_trunc_normal_
from models.base.backbone import BACKBONE_REGISTRY

# from models.base.base_blocks import (
#     STEM_REGISTRY, BRANCH_REGISTRY, HEAD_REGISTRY, DropPath, BaseHead
# )

def _cfg(url='', **kwargs):
    return {
        'url': url,
        'num_classes': 400, 'input_size': (3, 224, 224), 'pool_size': None,
        'crop_pct': .9, 'interpolation': 'bicubic',
        'mean': (0.5, 0.5, 0.5), 'std': (0.5, 0.5, 0.5),
        **kwargs
    }


class DropPath(nn.Module):
    """Drop paths (Stochastic Depth) per sample  (when applied in main path of residual blocks).
    """
    def __init__(self, drop_prob=None):
        super(DropPath, self).__init__()
        self.drop_prob = drop_prob

    def forward(self, x):
        return drop_path(x, self.drop_prob, self.training)
    
    def extra_repr(self) -> str:
        return 'p={}'.format(self.drop_prob)


class Mlp(nn.Module):
    def __init__(self, in_features, hidden_features=None, out_features=None, act_layer=nn.GELU, drop=0.):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.fc1 = nn.Linear(in_features, hidden_features)
        self.act = act_layer()
        self.fc2 = nn.Linear(hidden_features, out_features)
        self.drop = nn.Dropout(drop)

    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        # x = self.drop(x)
        # commit this for the orignal BERT implement 
        x = self.fc2(x)
        x = self.drop(x)
        return x


class AttentionImageNet(nn.Module):
    def __init__(self, dim, num_heads=8, qkv_bias=False, attn_drop=0., proj_drop=0.):
        super().__init__()
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = head_dim ** -0.5

        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

    def forward(self, x):
        B, N, C = x.shape
        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
        q, k, v = qkv.unbind(0)   # make torchscript happy (cannot use tensor as tuple)

        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)

        x = (attn @ v).transpose(1, 2).reshape(B, N, C)
        x = self.proj(x)
        x = self.proj_drop(x)
        return x


class Attention(nn.Module):
    def __init__(
            self, dim, num_heads=8, qkv_bias=False, qk_scale=None, attn_drop=0.,
            proj_drop=0., attn_head_dim=None):
        super().__init__()
        self.num_heads = num_heads
        head_dim = dim // num_heads
        if attn_head_dim is not None:
            head_dim = attn_head_dim
        all_head_dim = head_dim * self.num_heads
        self.scale = qk_scale or head_dim ** -0.5

        self.qkv = nn.Linear(dim, all_head_dim * 3, bias=False)
        if qkv_bias:
            self.q_bias = nn.Parameter(torch.zeros(all_head_dim))
            self.v_bias = nn.Parameter(torch.zeros(all_head_dim))
        else:
            self.q_bias = None
            self.v_bias = None

        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(all_head_dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

    def forward(self, x):
        B, N, C = x.shape
        qkv_bias = None
        if self.q_bias is not None:
            qkv_bias = torch.cat((self.q_bias, torch.zeros_like(self.v_bias, requires_grad=False), self.v_bias))
        # qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
        qkv = F.linear(input=x, weight=self.qkv.weight, bias=qkv_bias)
        qkv = qkv.reshape(B, N, 3, self.num_heads, -1).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]   # make torchscript happy (cannot use tensor as tuple)

        q = q * self.scale
        attn = (q @ k.transpose(-2, -1))

        
        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)

        x = (attn @ v).transpose(1, 2).reshape(B, N, -1)
        x = self.proj(x)
        x = self.proj_drop(x)
        return x


class Block(nn.Module):

    def __init__(self, dim, num_heads, mlp_ratio=4., qkv_bias=False, qk_scale=None, drop=0., attn_drop=0.,
                 drop_path=0., init_values=None, act_layer=nn.GELU, norm_layer=nn.LayerNorm,
                 attn_head_dim=None):
        super().__init__()
        self.norm1 = norm_layer(dim)
        self.attn = Attention(
            dim, num_heads=num_heads, qkv_bias=qkv_bias, qk_scale=qk_scale,
            attn_drop=attn_drop, proj_drop=drop, attn_head_dim=attn_head_dim)
        # NOTE: drop path for stochastic depth, we shall see if this is better than dropout here
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        self.norm2 = norm_layer(dim)
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = Mlp(in_features=dim, hidden_features=mlp_hidden_dim, act_layer=act_layer, drop=drop)

        if init_values > 0:
            self.gamma_1 = nn.Parameter(init_values * torch.ones((dim)),requires_grad=True)
            self.gamma_2 = nn.Parameter(init_values * torch.ones((dim)),requires_grad=True)
        else:
            self.gamma_1, self.gamma_2 = None, None

    def forward(self, x):
        if self.gamma_1 is None:
            x = x + self.drop_path(self.attn(self.norm1(x)))
            x = x + self.drop_path(self.mlp(self.norm2(x)))
        else:
            x = x + self.drop_path(self.gamma_1 * self.attn(self.norm1(x)))
            x = x + self.drop_path(self.gamma_2 * self.mlp(self.norm2(x)))
        return x


class BlockImageNet(nn.Module):

    def __init__(self, dim, num_heads, mlp_ratio=4., qkv_bias=False, qk_scale=None, drop=0., attn_drop=0.,
                 drop_path=0., init_values=None, act_layer=nn.GELU, norm_layer=nn.LayerNorm,
                 attn_head_dim=None):
        super().__init__()
        self.norm1 = norm_layer(dim)
        self.attn = AttentionImageNet(
            dim, num_heads=num_heads, qkv_bias=qkv_bias,
            attn_drop=attn_drop, proj_drop=drop)
        # NOTE: drop path for stochastic depth, we shall see if this is better than dropout here
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        self.norm2 = norm_layer(dim)
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = Mlp(in_features=dim, hidden_features=mlp_hidden_dim, act_layer=act_layer, drop=drop)

        if init_values > 0:
            self.gamma_1 = nn.Parameter(init_values * torch.ones((dim)),requires_grad=True)
            self.gamma_2 = nn.Parameter(init_values * torch.ones((dim)),requires_grad=True)
        else:
            self.gamma_1, self.gamma_2 = None, None

    def forward(self, x):
        if self.gamma_1 is None:
            x = x + self.drop_path(self.attn(self.norm1(x)))
            x = x + self.drop_path(self.mlp(self.norm2(x)))
        else:
            x = x + self.drop_path(self.gamma_1 * self.attn(self.norm1(x)))
            x = x + self.drop_path(self.gamma_2 * self.mlp(self.norm2(x)))
        return x


class PatchEmbed(nn.Module):
    """ Image to Patch Embedding
    """
    def __init__(self, img_size=224, patch_size=16, in_chans=3, embed_dim=768, num_frames=16, tubelet_size=2):
        super().__init__()
        img_size = to_2tuple(img_size)
        patch_size = to_2tuple(patch_size)
        self.tubelet_size = int(tubelet_size)
        num_patches = (img_size[1] // patch_size[1]) * (img_size[0] // patch_size[0]) * (num_frames // self.tubelet_size)
        self.img_size = img_size
        self.patch_size = patch_size
        self.num_patches = num_patches
        self.proj = nn.Conv3d(in_channels=in_chans, out_channels=embed_dim, 
                            kernel_size = (self.tubelet_size,  patch_size[0],patch_size[1]), 
                            stride=(self.tubelet_size,  patch_size[0],  patch_size[1]))

    def forward(self, x, **kwargs):
        B, C, T, H, W = x.shape
        # FIXME look at relaxing size constraints
        assert H == self.img_size[0] and W == self.img_size[1], \
            f"Input image size ({H}*{W}) doesn't match model ({self.img_size[0]}*{self.img_size[1]})."
        x = self.proj(x).flatten(2).transpose(1, 2)
        return x
    
# sin-cos position encoding
# https://github.com/jadore801120/attention-is-all-you-need-pytorch/blob/master/transformer/Models.py#L31
def get_sinusoid_encoding_table(n_position, d_hid): 
    ''' Sinusoid position encoding table ''' 
    # TODO: make it with torch instead of numpy 
    def get_position_angle_vec(position): 
        return [position / np.power(10000, 2 * (hid_j // 2) / d_hid) for hid_j in range(d_hid)] 

    sinusoid_table = np.array([get_position_angle_vec(pos_i) for pos_i in range(n_position)]) 
    sinusoid_table[:, 0::2] = np.sin(sinusoid_table[:, 0::2]) # dim 2i 
    sinusoid_table[:, 1::2] = np.cos(sinusoid_table[:, 1::2]) # dim 2i+1 

    return torch.FloatTensor(sinusoid_table).unsqueeze(0) 





def trunc_normal_(tensor, mean=0., std=1.):
    __call_trunc_normal_(tensor, mean=mean, std=std, a=-std, b=std)


__all__ = [
    'pretrain_videomae_base_patch16_224', 
    'pretrain_videomae_large_patch16_224', 
]


@BACKBONE_REGISTRY.register()
class VitVideoEncoder(nn.Module):
    """ Vision Transformer with support for patch or hybrid CNN input stage
    """
    def __init__(self, cfg):
        super().__init__()
        img_size = cfg.DATA.TRAIN_CROP_SIZE
        patch_size = cfg.VIDEO.BACKBONE.PATCH_SIZE
        in_chans = 3
        num_classes = 0
        embed_dim = cfg.VIDEO.BACKBONE.NUM_FEATURES
        depth = cfg.VIDEO.BACKBONE.DEPTH
        num_heads = cfg.VIDEO.BACKBONE.NUM_HEADS
        mlp_ratio = cfg.VIDEO.BACKBONE.MLP_MULT
        qkv_bias = True
        qk_scale = None
        drop_rate = cfg.VIDEO.BACKBONE.FF_DROPOUT
        attn_drop_rate = cfg.VIDEO.BACKBONE.ATTN_DROPOUT
        drop_path_rate = cfg.VIDEO.BACKBONE.DROP_PATH
        use_cls_token = cfg.VIDEO.BACKBONE.USE_CLS_TOKEN
        freeze_conv1 = cfg.VIDEO.BACKBONE.FREEZE_CONV1
        norm_layer = partial(nn.LayerNorm, eps=1e-6)
        init_values = 0.0
        tubelet_size = cfg.VIDEO.BACKBONE.TUBELET_SIZE
        self.use_learnable_pos_emb = cfg.VIDEO.BACKBONE.POS_EMBD_LEARNABLE
        self.final_norm = cfg.VIDEO.BACKBONE.FINAL_NORM
        self.num_classes = num_classes
        self.num_features = self.embed_dim = embed_dim  # num_features for consistency with other models
        self.patch_embed = PatchEmbed(
            img_size=img_size, patch_size=patch_size, in_chans=in_chans, embed_dim=embed_dim,tubelet_size=tubelet_size, num_frames=cfg.DATA.NUM_INPUT_FRAMES)
        num_patches = self.patch_embed.num_patches

        # TODO: Add the cls token
        if use_cls_token:
            assert self.use_learnable_pos_emb
            self.cls_token          = nn.Parameter(torch.randn(1, 1, embed_dim))
            self.pos_embed = nn.Parameter(torch.zeros(1, num_patches + 1, embed_dim))
        else:
            self.cls_token = None
            if self.use_learnable_pos_emb:
                self.pos_embed = nn.Parameter(torch.zeros(1, num_patches, embed_dim))
            else:
                # sine-cosine positional embeddings 
                self.pos_embed = get_sinusoid_encoding_table(num_patches, embed_dim)

        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, depth)]  # stochastic depth decay rule
        if hasattr(cfg.VIDEO.BACKBONE, "BLOCK_TYPE") and cfg.VIDEO.BACKBONE.BLOCK_TYPE == 'image':
            block_module = BlockImageNet
        else:
            block_module = Block
        self.blocks = nn.ModuleList([
            block_module(
                dim=embed_dim, num_heads=num_heads, mlp_ratio=mlp_ratio, qkv_bias=qkv_bias, qk_scale=qk_scale,
                drop=drop_rate, attn_drop=attn_drop_rate, drop_path=dpr[i], norm_layer=norm_layer,
                init_values=init_values)
            for i in range(depth)])
        if self.final_norm:
            self.norm = norm_layer(embed_dim)
        else:
            self.norm = nn.Identity()
        self.head = nn.Linear(embed_dim, num_classes) if num_classes > 0 else nn.Identity()

        if self.use_learnable_pos_emb:
            trunc_normal_(self.pos_embed, std=.02)

        self.apply(self._init_weights)
        if freeze_conv1:
            self.patch_embed.proj.weight.requires_grad = False
            self.patch_embed.proj.bias.requires_grad = False

    def _init_weights(self, m):
        if self.cls_token is not None:
            nn.init.normal_(self.cls_token, std=1e-6)
        # weight initialization
        for name, m in self.named_modules():
            if isinstance(m, nn.Linear):
                if 'qkv' in name:
                    # treat the weights of Q, K, V separately
                    val = math.sqrt(6. / float(m.weight.shape[0] // 3 + m.weight.shape[1]))
                    nn.init.uniform_(m.weight, -val, val)
                else:
                    nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
        if isinstance(self.patch_embed, PatchEmbed):
            # xavier_uniform initialization
            from functools import reduce
            from operator import mul
            val = math.sqrt(6. / float(3 * reduce(mul, self.patch_embed.patch_size, 1) + self.patch_embed.proj.weight.shape[0]))
            nn.init.uniform_(self.patch_embed.proj.weight, -val, val)
            nn.init.zeros_(self.patch_embed.proj.bias)
        # if isinstance(m, nn.Linear):
        #     nn.init.xavier_uniform_(m.weight)
        #     if isinstance(m, nn.Linear) and m.bias is not None:
        #         nn.init.constant_(m.bias, 0)
        # elif isinstance(m, nn.LayerNorm):
        #     nn.init.constant_(m.bias, 0)
        #     nn.init.constant_(m.weight, 1.0)

    def get_num_layers(self):
        return len(self.blocks), 0

    @torch.jit.ignore
    def no_weight_decay(self):
        return {'pos_embed', 'cls_token'}

    def get_classifier(self):
        return self.head

    def reset_classifier(self, num_classes, global_pool=''):
        self.num_classes = num_classes
        self.head = nn.Linear(self.embed_dim, num_classes) if num_classes > 0 else nn.Identity()

    def forward_features(self, x):
        _, _, T, _, _ = x.shape
        patch_embed = self.patch_embed(x)
        if self.cls_token is not None:
            cls_token = self.cls_token.repeat((x.shape[0],1,1))
            patch_embed = torch.cat((cls_token, patch_embed), dim = 1)
        if self.use_learnable_pos_emb:
            x = patch_embed + self.pos_embed.type_as(patch_embed).to(patch_embed.device).clone()
        else:
            x = patch_embed + self.pos_embed.type_as(patch_embed).to(patch_embed.device).clone().detach()

        B, _, C = x.shape
        for blk in self.blocks:
            x = blk(x)

        if self.cls_token is not None:
            x = self.norm(x)[:, 0, :]
        else:
            x = self.norm(x).mean(dim=1)
        return x

    def forward(self, x):
        if type(x) is list:
            x = x[0]
        elif isinstance(x, dict):
            x = x["video"]
        x = self.forward_features(x)
        x = self.head(x)
        return x


@BACKBONE_REGISTRY.register()
class VitVideoMAEEncoder(VitVideoEncoder):
    """ Vision Transformer with support for patch or hybrid CNN input stage
    """
    def __init__(self, cfg):
        super().__init__(cfg)
    
    def forward_features(self, x, mask):
        _, _, T, _, _ = x.shape
        patch_embed = self.patch_embed(x)
        x = patch_embed + self.pos_embed.type_as(patch_embed).to(patch_embed.device).clone().detach()
        x = x[mask, :].view(patch_embed.size(0), -1, patch_embed.size(-1))
        B, _, C = x.shape
        for blk in self.blocks:
            x = blk(x)
        return x

    def forward(self, x, mask):
        x = self.forward_features(x, mask)
        x = self.head(x)
        return x


@BACKBONE_REGISTRY.register()
class VitVideoMAEDecoder(nn.Module):
    """ Vision Transformer with support for patch or hybrid CNN input stage
    """
    def __init__(self, cfg, encoder):
        super().__init__()

        encoder_embed_dim = cfg.VIDEO.BACKBONE.NUM_FEATURES
        decoder_embed_dim = 384
        decoder_num_heads = 6
        decoder_depth = 4
        drop_rate = 0
        attn_drop_rate = 0
        mlp_ratio = 4
        qkv_bias = True
        qk_scale = None
        init_values = 0.0
        norm_layer = partial(nn.LayerNorm, eps=1e-6)
        num_patches = encoder.patch_embed.num_patches
        patch_size = encoder.patch_embed.patch_size[0]
        tubelet_size = cfg.VIDEO.BACKBONE.TUBELET_SIZE
        self.encoder_to_decoder = nn.Linear(encoder_embed_dim, decoder_embed_dim, bias=False)
        self.mask_token = nn.Parameter(torch.zeros(1, 1, decoder_embed_dim))
        self.pos_embed  = get_sinusoid_encoding_table(num_patches, decoder_embed_dim)
        self.blocks = nn.ModuleList([
            Block(
                dim=decoder_embed_dim, num_heads=decoder_num_heads, mlp_ratio=mlp_ratio, qkv_bias=qkv_bias, qk_scale=qk_scale,
                drop=drop_rate, attn_drop=attn_drop_rate, drop_path=0, norm_layer=norm_layer,
                init_values=init_values)
            for i in range(decoder_depth)])
        self.head = nn.Linear(decoder_embed_dim, 3 * tubelet_size * patch_size ** 2 )
        self.apply(self._init_weights)

    def _init_weights(self, m):
        # weight initialization
        for name, m in self.named_modules():
            if isinstance(m, nn.Linear):
                if 'qkv' in name:
                    # treat the weights of Q, K, V separately
                    val = math.sqrt(6. / float(m.weight.shape[0] // 3 + m.weight.shape[1]))
                    nn.init.uniform_(m.weight, -val, val)
                else:
                    nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def forward(self, x, mask):
        x = self.encoder_to_decoder(x)
        tokens = self.mask_token.repeat(mask.size(0), mask.size(1), 1)
        tokens[mask] = x.flatten(0, 1)
        tokens = tokens + self.pos_embed.type_as(tokens).to(tokens.device).clone().detach()
        for block in self.blocks:
            tokens = block(tokens)
        pre_pixels = self.head(tokens)
        pre_pixels = pre_pixels[~mask, :].view(mask.size(0), -1, pre_pixels.size(-1))
        return pre_pixels