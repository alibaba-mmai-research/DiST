#!/usr/bin/env python3
# Copyright (C) Alibaba Group Holding Limited. 

import torch
import utils.distributed as du

def get_sim_func(func_name, pair='pos'):
    assert pair in ('pos', 'neg')
    func = "sim_func_" + func_name
    pair_specific_func = func + '_' + pair
    if pair_specific_func not in globals() and func not in globals():
        raise NotImplementedError("Unknown similarity function: {}".format(func_name))
    elif pair_specific_func in globals():
        return globals()[pair_specific_func]
    else:
        return globals()[func]

def sim_func_linear(sim, temperature, optim_target=None):
    if optim_target is not None:
        return torch.exp(
            sim.clamp(-1, optim_target)/temperature
        )
    else:
        return torch.exp(
            sim/temperature
        )

def sim_func_parabola_pos(sim, temperature, optim_target=1.0):
    return torch.exp(
        (1-(sim-optim_target)**2)/temperature
    )

def sim_func_parabola_neg(sim, temperature):
    return torch.exp(
        (sim+1)**2/temperature
    )

def contrastive_instance_discrimination(cfg, logits, batch_size, samples):
    device = logits.device

    mask_ins = torch.eye(batch_size, device=device).repeat_interleave(samples, dim=1).repeat_interleave(samples, dim=0)
    pos_mask = 1-torch.eye(batch_size*samples, device=device)
    
    sim = torch.matmul(logits, logits.transpose(0,1))
    pos_sim_mtx = get_sim_func(cfg.PRETRAIN.CONTRASTIVE.SIM_FUNC_POS)(
        sim, cfg.PRETRAIN.CONTRASTIVE.TEMPERATURE, cfg.PRETRAIN.CONTRASTIVE.POS_OPTIM_TARGET
    )
    neg_sim_mtx = get_sim_func(cfg.PRETRAIN.CONTRASTIVE.SIM_FUNC_NEG)(
        sim, cfg.PRETRAIN.CONTRASTIVE.TEMPERATURE
    )

    if cfg.PRETRAIN.CONTRASTIVE.INS_MIL:
        pos = pos_sim_mtx[(mask_ins*pos_mask)!=0].reshape(-1, samples-1).sum(1, keepdim=True)
    else:
        pos = pos_sim_mtx[(mask_ins*pos_mask)!=0].reshape(-1, samples-1)
    neg = ((1-mask_ins)*neg_sim_mtx).sum(0).unsqueeze(1)

    if cfg.PRETRAIN.CONTRASTIVE.WITH_ONE:
        N = pos.shape[1]
        loss = -((1/N) * torch.log(pos/(pos+neg)).sum()) / (batch_size*samples)
        return loss, pos.mean().item(), neg.mean().item()
    else:
        N = pos.shape[1]
        return -((1/N) * torch.log(pos/neg).sum()) / (batch_size*samples)        


def contrastive_augmentation_discrimination(cfg, logits, batch_size, samples):
    device = logits.device

    mask_aug = torch.eye(samples, device=device).repeat(batch_size, batch_size)
    pos_mask = 1-torch.eye(batch_size*samples, device=device)
    
    sim = torch.matmul(logits, logits.transpose(0,1))
    pos_sim_mtx = get_sim_func(cfg.PRETRAIN.CONTRASTIVE.SIM_FUNC_POS)(
        sim, cfg.PRETRAIN.CONTRASTIVE.TEMPERATURE, cfg.PRETRAIN.CONTRASTIVE.POS_OPTIM_TARGET
    )
    neg_sim_mtx = get_sim_func(cfg.PRETRAIN.CONTRASTIVE.SIM_FUNC_NEG)(
        sim, cfg.PRETRAIN.CONTRASTIVE.TEMPERATURE
    )

    if cfg.PRETRAIN.CONTRASTIVE.AUG_MIL:
        pos = pos_sim_mtx[(mask_aug*pos_mask)!=0].reshape(-1, batch_size-1).sum(1, keepdim=True)
    else:
        pos = pos_sim_mtx[(mask_aug*pos_mask)!=0].reshape(-1, batch_size-1)
    neg = ((1-mask_aug)*neg_sim_mtx).sum(0).unsqueeze(1)

    if cfg.PRETRAIN.CONTRASTIVE.WITH_ONE:
        N = pos.shape[1]
        return -((1/N) * torch.log(pos/(pos+neg)).sum()) / (batch_size*samples)
    else:
        N = pos.shape[1]
        return -((1/N) * torch.log(pos/neg).sum()) / (batch_size*samples)   

def aug_ins_space_constraint(cfg, logits_aug, logits_ins):
    device = logits_aug.device
    sim_aug_ins = torch.matmul(logits_ins, logits_aug.transpose(0,1))
    return 0


def contrastive_hico(cfg, preds, logits, batch_size, samples):
    device = logits.device
    mask_ins = torch.eye(batch_size, device=device, dtype=torch.bool).repeat_interleave(samples, dim=1).repeat_interleave(samples, dim=0)
    pos_mask = ~torch.eye(batch_size*samples, device=device, dtype=torch.bool)

    sim_vcl = torch.matmul(logits, logits.transpose(0,1))
    pos_sim_mtx_vcl = get_sim_func(cfg.PRETRAIN.CONTRASTIVE.SIM_FUNC_POS)(
        sim_vcl, cfg.PRETRAIN.CONTRASTIVE.TEMPERATURE, cfg.PRETRAIN.CONTRASTIVE.POS_OPTIM_TARGET
    )
    neg_sim_mtx_vcl = get_sim_func(cfg.PRETRAIN.CONTRASTIVE.SIM_FUNC_NEG)(
        sim_vcl, cfg.PRETRAIN.CONTRASTIVE.TEMPERATURE
    )

    vcl_mask = torch.tensor([1, 1] + [0 for i in range(samples-2)], device=device, dtype=torch.bool).repeat(batch_size)
    pos_vcl = pos_sim_mtx_vcl[mask_ins&pos_mask].reshape(-1, samples-1)

    neg_vcl = ((1-mask_ins.float())*neg_sim_mtx_vcl).sum(0).unsqueeze(1)

    vcl_pos, vcl_neg = pos_vcl[vcl_mask, :1], neg_vcl[vcl_mask, :]

    N_vcl = vcl_pos.size(1)
    vcl_loss = -((1/N_vcl) * (torch.log(vcl_pos/(vcl_pos+vcl_neg))).sum()) / (vcl_mask.sum())
    # tcl_mask = ~vcl_mask

    pos_sim_mtx_tcl = preds.sigmoid()

    neg_sim_mtx_tcl = pos_sim_mtx_tcl
    rank = du.get_rank()
    mask_s, mask_e = rank*pos_sim_mtx_tcl.size(0), (rank+1)*pos_sim_mtx_tcl.size(0)
    # tcl_mask = tcl_mask[mask_s:mask_e]
    mask_ins = torch.eye(pos_sim_mtx_tcl.size(0)//samples, device=device, dtype=torch.bool).repeat_interleave(samples, dim=1).repeat_interleave(samples, dim=0)
    pos_mask = ~torch.eye(pos_sim_mtx_tcl.size(0), device=device, dtype=torch.bool)

    pos_tcl = pos_sim_mtx_tcl[mask_ins&pos_mask].reshape(-1)
    neg_tcl = neg_sim_mtx_tcl[~mask_ins].reshape(-1)

    log_eps = 1e-5
    gama = cfg.HICO.LOSS.GAMA
    tcl_loss = -(torch.pow((1.0 - pos_tcl), gama) * torch.log(pos_tcl+log_eps)).mean() - (torch.pow((neg_tcl), gama) * torch.log(1.0-neg_tcl+log_eps)).mean()
    
    loss = vcl_loss * cfg.HICO.LOSS.VCL_WEIGHT * du.get_world_size() + tcl_loss * cfg.HICO.LOSS.TCL_WEIGHT
    return loss, vcl_pos.mean().item(), vcl_neg.mean().item(), vcl_loss.item(), tcl_loss.item()


def contrastive_hico_plus_plus(cfg, preds, logits, batch_size, samples):
    device = logits.device
    mask_ins = torch.eye(batch_size * samples // 2, device=device, dtype=torch.bool).repeat_interleave(2, dim=1).repeat_interleave(2, dim=0)
    pos_mask = ~torch.eye(batch_size*samples, device=device, dtype=torch.bool)

    sim_vcl = torch.matmul(logits, logits.transpose(0,1))
    pos_sim_mtx_vcl = get_sim_func(cfg.PRETRAIN.CONTRASTIVE.SIM_FUNC_POS)(
        sim_vcl, cfg.PRETRAIN.CONTRASTIVE.TEMPERATURE, cfg.PRETRAIN.CONTRASTIVE.POS_OPTIM_TARGET
    )
    neg_sim_mtx_vcl = get_sim_func(cfg.PRETRAIN.CONTRASTIVE.SIM_FUNC_NEG)(
        sim_vcl, cfg.PRETRAIN.CONTRASTIVE.TEMPERATURE
    )

    pos_vcl = pos_sim_mtx_vcl[mask_ins&pos_mask].reshape(-1, 2 - 1)

    mask_ins_neg = torch.eye(batch_size, device=device, dtype=torch.bool).repeat_interleave(samples, dim=1).repeat_interleave(samples, dim=0)
    neg_vcl = ((1-mask_ins_neg.float())*neg_sim_mtx_vcl).sum(0).unsqueeze(1)

    vcl_pos, vcl_neg = pos_vcl, neg_vcl

    N_vcl = vcl_pos.size(1)

    vcl_loss = -((1/N_vcl) * (torch.log(vcl_pos/(vcl_pos+vcl_neg))).mean())
    # tcl_mask = ~vcl_mask
    pos_sim_mtx_tcl = preds.sigmoid()

    neg_sim_mtx_tcl = pos_sim_mtx_tcl
    rank = du.get_rank()
    mask_s, mask_e = rank*pos_sim_mtx_tcl.size(0), (rank+1)*pos_sim_mtx_tcl.size(0)
    # tcl_mask = tcl_mask[mask_s:mask_e]
    mask_ins = torch.eye(pos_sim_mtx_tcl.size(0)//(samples // 2), device=device, dtype=torch.bool).repeat_interleave(samples//2, dim=1).repeat_interleave(samples//2, dim=0)
    pos_mask = ~torch.eye(pos_sim_mtx_tcl.size(0), device=device, dtype=torch.bool)

    pos_tcl = pos_sim_mtx_tcl[mask_ins&pos_mask].reshape(-1)
    neg_tcl = neg_sim_mtx_tcl[~mask_ins].reshape(-1)

    log_eps = 1e-5
    gama = cfg.HICO.LOSS.GAMA
    tcl_loss = -(torch.pow((1.0 - pos_tcl), gama) * torch.log(pos_tcl+log_eps)).mean() - (torch.pow((neg_tcl), gama) * torch.log(1.0-neg_tcl+log_eps)).mean()
    
    loss = vcl_loss * cfg.HICO.LOSS.VCL_WEIGHT * du.get_world_size() + tcl_loss * cfg.HICO.LOSS.TCL_WEIGHT
    return loss, vcl_pos.mean().item(), vcl_neg.mean().item(), vcl_loss.item(), tcl_loss.item()


def contrastive_hico_plus_plus_vit(cfg, preds, logits, batch_size, samples):
    device = logits.device
    mask_ins = torch.eye(batch_size * samples // 2, device=device, dtype=torch.bool).repeat_interleave(2, dim=1).repeat_interleave(2, dim=0)
    pos_mask = ~torch.eye(batch_size*samples, device=device, dtype=torch.bool)

    sim_vcl = torch.matmul(logits, logits.transpose(0,1))
    pos_sim_mtx_vcl = get_sim_func(cfg.PRETRAIN.CONTRASTIVE.SIM_FUNC_POS)(
        sim_vcl, cfg.PRETRAIN.CONTRASTIVE.TEMPERATURE, cfg.PRETRAIN.CONTRASTIVE.POS_OPTIM_TARGET
    )
    neg_sim_mtx_vcl = get_sim_func(cfg.PRETRAIN.CONTRASTIVE.SIM_FUNC_NEG)(
        sim_vcl, cfg.PRETRAIN.CONTRASTIVE.TEMPERATURE
    )

    pos_vcl = pos_sim_mtx_vcl[mask_ins&pos_mask].reshape(-1, 2 - 1)

    mask_ins_neg = torch.eye(batch_size, device=device, dtype=torch.bool).repeat_interleave(samples, dim=1).repeat_interleave(samples, dim=0)
    neg_vcl = ((1-mask_ins_neg.float())*neg_sim_mtx_vcl).sum(0).unsqueeze(1)

    vcl_pos, vcl_neg = pos_vcl, neg_vcl

    N_vcl = vcl_pos.size(1)
    vcl_loss = -((1/N_vcl) * (torch.log(vcl_pos/(vcl_pos+vcl_neg))).mean())
    vcl_loss = vcl_loss * cfg.PRETRAIN.CONTRASTIVE.TEMPERATURE * 2
    # tcl_mask = ~vcl_mask
    pos_sim_mtx_tcl = preds.sigmoid()

    neg_sim_mtx_tcl = pos_sim_mtx_tcl
    rank = du.get_rank()
    mask_s, mask_e = rank*pos_sim_mtx_tcl.size(0), (rank+1)*pos_sim_mtx_tcl.size(0)
    # tcl_mask = tcl_mask[mask_s:mask_e]
    mask_ins = torch.eye(pos_sim_mtx_tcl.size(0)//(samples // 2), device=device, dtype=torch.bool).repeat_interleave(samples//2, dim=1).repeat_interleave(samples//2, dim=0)
    pos_mask = ~torch.eye(pos_sim_mtx_tcl.size(0), device=device, dtype=torch.bool)

    pos_tcl = pos_sim_mtx_tcl[mask_ins&pos_mask].reshape(-1)
    neg_tcl = neg_sim_mtx_tcl[~mask_ins].reshape(-1)

    log_eps = 1e-5
    gama = cfg.HICO.LOSS.GAMA
    tcl_loss = -(torch.pow((1.0 - pos_tcl), gama) * torch.log(pos_tcl+log_eps)).mean() - (torch.pow((neg_tcl), gama) * torch.log(1.0-neg_tcl+log_eps)).mean()
    
    loss = vcl_loss * cfg.HICO.LOSS.VCL_WEIGHT * du.get_world_size() + tcl_loss * cfg.HICO.LOSS.TCL_WEIGHT
    return loss, vcl_pos.mean().item(), vcl_neg.mean().item(), vcl_loss.item(), tcl_loss.item()

