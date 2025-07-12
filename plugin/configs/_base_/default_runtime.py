# MMEngine-style default runtime configuration

default_scope = 'mmdet'

# Default hooks that runner will automatically register
default_hooks = dict(
    timer=dict(type='IterTimerHook'),
    logger=dict(type='LoggerHook', interval=50),
    param_scheduler=dict(type='ParamSchedulerHook'),
    checkpoint=dict(type='CheckpointHook', interval=1),
    sampler_seed=dict(type='DistSamplerSeedHook')
)

# Distributed environment configuration
env_cfg = dict(dist_cfg=dict(backend='nccl'))

# Logging & misc
log_level = 'INFO'
load_from = None
resume_from = None
resume = False
work_dir = None

# Keep workflow definition for compatibility
workflow = [('train', 1)]

# ---- Optimizer (MMEngine-style) ---------------------------------
optim_wrapper = dict(
    type='OptimWrapper',
    optimizer=dict(
        type='AdamW',
        lr=5e-4,
        weight_decay=1e-2,
    ),
    # per-parameter rules or grad clipping can be overridden in child configs
)
