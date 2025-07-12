# ---------------------------------------------
# Copyright (c) OpenMMLab. All rights reserved.
# ---------------------------------------------
#  Modified by Zhiqi Li
# ---------------------------------------------
#  Modified by Shihao Wang
# ---------------------------------------------
from mmengine.registry import Registry
from mmdet.registry import DATA_SAMPLERS as SAMPLER

# SAMPLER = Registry('sampler')

def build_sampler(cfg, default_args):
    return SAMPLER.build(cfg, default_args=default_args)