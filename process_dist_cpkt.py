import os
import torch


import collections

root = '/mnt/data/code/open-source/open-source-hico-tpami/HiCo/output/open-source-cpkts/'
files = os.listdir(root)

def rename_model_state(model_state):
    new_dict = collections.OrderedDict()
    for name, param in model_state.items():
        if 'ladder_net' in name:
            name = name.replace("ladder_net.temporal_stem", "dist_net.temporal_stem")
            name = name.replace("ladder_net.input_map_feat_nets", "dist_net.input_linears")
            name = name.replace("ladder_net.s2t_fuse_nets", "dist_net.integration2temporal_nets")
            name = name.replace("ladder_net.t2s_fuse_nets", "dist_net.temporal2integration_nets")
            name = name.replace("ladder_net.temporal_nets", "dist_net.temporal_nets")
            name = name.replace("ladder_net.spatial_nets", "dist_net.integration_nets")
            name = name.replace("ladder_net.final_temporal_nets", "dist_net.adapooling_nets")
            name = name.replace("ladder_net.proj_spatial_cls_token", "dist_net.proj_spatial_cls_token")
            name = name.replace("ladder_net.ln_post", "dist_net.ln_post")
            name = name.replace("ladder_net.proj", "dist_net.proj")
            name = name.replace("ladder_net.aggregated_cls_token", "dist_net.aggregated_cls_token")
            name = name.replace("ladder_net.aggregated_spatial_cls_token", "dist_net.aggregated_spatial_cls_token")
            if "ladder_net" in name:
                print(name)
            assert "ladder_net" not in name
        new_dict[name] = param
    return new_dict


for f in files:
    if not f.startswith("DIST"):
        continue
    d = torch.load(os.path.join(root, f))
    if len(d.keys()) > 1:
        torch.save({"model_state": rename_model_state(d["model_state"])}, os.path.join(root, f))
