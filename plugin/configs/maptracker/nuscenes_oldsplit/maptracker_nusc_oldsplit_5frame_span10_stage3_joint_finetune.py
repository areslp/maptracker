_base_ = [
    '../../_base_/default_runtime.py'
]

default_scope = 'mmdet'

# model type
type = 'Mapper'
plugin = True

# plugin code dir
plugin_dir = 'plugin/'
[]
# img configs
img_norm_cfg = dict(
    mean=[103.530, 116.280, 123.675], std=[1.0, 1.0, 1.0], to_rgb=False)

img_h = 480
img_w = 800
img_size = (img_h, img_w)

num_gpus = 8
batch_size = 2
num_iters_per_epoch = 27968 // (num_gpus * batch_size)
num_epochs = 48
num_epochs_interval = num_epochs // 8
total_iters = num_epochs * num_iters_per_epoch
num_queries = 100

# category configs
cat2id = {
    'ped_crossing': 0,
    'divider': 1,
    'boundary': 2,
}
num_class = max(list(cat2id.values())) + 1

# bev configs
roi_size = (60, 30) # bev range, 60m in x-axis, 30m in y-axis
bev_h = 50
bev_w = 100
pc_range = [-roi_size[0]/2, -roi_size[1]/2, -3, roi_size[0]/2, roi_size[1]/2, 5]

# vectorize params
coords_dim = 2
sample_dist = -1
sample_num = -1
simplify = True

# rasterize params (for temporal matching use)
canvas_size = (200, 100) # bev feature size
thickness = 3 # thickness of rasterized polylines

# meta info for submission pkl
meta = dict(
    use_lidar=False,
    use_camera=True,
    use_radar=False,
    use_map=False,
    use_external=False,
    output_format='vector')

# model configs
bev_embed_dims = 256
embed_dims = 512
num_feat_levels = 3
norm_cfg = dict(type='BN2d')
num_class = max(list(cat2id.values()))+1
num_points = 20
permute = True

model = dict(
    type='MapTracker',
    roi_size=roi_size,
    bev_h=bev_h,
    bev_w=bev_w,
    history_steps=4,
    test_time_history_steps=20,
    mem_select_dist_ranges=[1, 5, 10, 15],
    skip_vector_head=False,
    freeze_bev=False,
    track_fp_aug=False,
    use_memory=True,
    mem_len=4,
    mem_warmup_iters=-1,
    backbone_cfg=dict(
        type='BEVFormerBackbone',
        roi_size=roi_size,
        bev_h=bev_h,
        bev_w=bev_w,
        history_steps=4,
        use_grid_mask=True,
        img_backbone=dict(
            type='ResNet',
            with_cp=False,
            # pretrained='./resnet50_checkpoint.pth',
            pretrained='open-mmlab://detectron2/resnet50_caffe',
            depth=50,
            num_stages=4,
            out_indices=(1, 2, 3),
            frozen_stages=-1,
            norm_cfg=norm_cfg,
            norm_eval=True,
            style='caffe',
            dcn=dict(type='DCNv2', deform_groups=1, fallback_on_stride=False),
            stage_with_dcn=(False, False, True, True)
            ),
        img_neck=dict(
            type='FPN',
            in_channels=[512, 1024, 2048],
            out_channels=bev_embed_dims,
            start_level=0,
            add_extra_convs=True,
            num_outs=num_feat_levels,
            norm_cfg=norm_cfg,
            relu_before_extra_convs=True),
        transformer=dict(
            type='PerceptionTransformer',
            embed_dims=bev_embed_dims,
            encoder=dict(
                type='BEVFormerEncoder',
                num_layers=2,
                pc_range=pc_range,
                num_points_in_pillar=4,
                return_intermediate=False,
                transformerlayers=dict(
                    type='BEVFormerLayer',
                    attn_cfgs=[
                        dict(
                            type='TemporalSelfAttention',
                            embed_dims=bev_embed_dims,
                            num_levels=1),
                        dict(
                            type='SpatialCrossAttention',
                            deformable_attention=dict(
                                type='MSDeformableAttention3D',
                                embed_dims=bev_embed_dims,
                                num_points=8,
                                num_levels=num_feat_levels),
                            embed_dims=bev_embed_dims,
                        )
                    ],
                    feedforward_channels=bev_embed_dims*2,
                    ffn_dropout=0.1,
                    operation_order=('self_attn', 'norm', 'cross_attn', 'norm',
                                    'ffn', 'norm')
                )
            ),
        ),
        positional_encoding=dict(
            type='LearnedPositionalEncoding',
            num_feats=bev_embed_dims//2,
            row_num_embed=bev_h,
            col_num_embed=bev_w,
            ),
    ),
    head_cfg=dict(
        type='MapDetectorHead',
        num_queries=num_queries,
        embed_dims=embed_dims,
        num_classes=num_class,
        in_channels=bev_embed_dims,
        num_points=num_points,
        roi_size=roi_size,
        coord_dim=2,
        different_heads=False,
        predict_refine=False,
        sync_cls_avg_factor=True,
        trans_loss_weight=0.1,
        transformer=dict(
            type='MapTransformer',
            num_feature_levels=1,
            num_points=num_points,
            coord_dim=2,
            encoder=dict(
                type='PlaceHolderEncoder',
                embed_dims=embed_dims,
            ),
            decoder=dict(
                type='MapTransformerDecoder_new',
                num_layers=6,
                prop_add_stage=1,
                return_intermediate=True,
                transformerlayers=dict(
                    type='MapTransformerLayer',
                    attn_cfgs=[
                        dict(
                            type='MultiheadAttention',
                            embed_dims=embed_dims,
                            num_heads=8,
                            attn_drop=0.1,
                            proj_drop=0.1,
                        ),
                        dict(
                            type='CustomMSDeformableAttention',
                            embed_dims=embed_dims,
                            num_heads=8,
                            num_levels=1,
                            num_points=num_points,
                            dropout=0.1,
                        ),
                        dict(
                            type='MultiheadAttention',
                            embed_dims=embed_dims,
                            num_heads=8,
                            attn_drop=0.1,
                            proj_drop=0.1,
                        ),
                    ],
                    ffn_cfgs=dict(
                        type='FFN',
                        embed_dims=embed_dims,
                        feedforward_channels=embed_dims*2,
                        num_fcs=2,
                        ffn_drop=0.1,
                        act_cfg=dict(type='ReLU', inplace=True),        
                    ),
                    feedforward_channels=embed_dims*2,
                    ffn_dropout=0.1,
                    ## an addtional cross attention for vector memory fusion
                    operation_order=('self_attn', 'norm', 'cross_attn', 'norm', 'cross_attn', 'norm',
                                    'ffn', 'norm')
                )
            )
        ),
        loss_cls=dict(
            type='FocalLoss',
            use_sigmoid=True,
            gamma=2.0,
            alpha=0.25,
            loss_weight=5.0
        ),
        loss_reg=dict(
            type='LinesL1Loss',
            loss_weight=50.0,
            beta=0.01,
        ),
        assigner=dict(
            type='HungarianLinesAssigner',
                cost=dict(
                    type='MapQueriesCost',
                    cls_cost=dict(type='FocalLossCost', weight=5.0),
                    reg_cost=dict(type='LinesL1Cost', weight=50.0, beta=0.01, permute=permute),
                    ),
                ),
        ),
    seg_cfg=dict(
        type='MapSegHead',
        num_classes=num_class,
        in_channels=bev_embed_dims,
        embed_dims=bev_embed_dims,
        bev_size=(bev_w, bev_h),
        canvas_size=canvas_size,
        loss_seg=dict(
            type='MaskFocalLoss',
            use_sigmoid=True,
            loss_weight=10.0,
        ),
        loss_dice=dict(
            type='MaskDiceLoss',
            loss_weight=1.0,
        )
    ),
    model_name='SingleStage'
)

# data processing pipelines
train_pipeline = [
    dict(
        type='VectorizeMap',
        coords_dim=coords_dim,
        roi_size=roi_size,
        sample_num=num_points,
        normalize=True,
        permute=permute,
    ),
    dict(
        type='RasterizeMap',   
        roi_size=roi_size,
        coords_dim=coords_dim,
        canvas_size=canvas_size,
        thickness=thickness,
        semantic_mask=True,
    ),
    dict(type='LoadMultiViewImagesFromFiles', to_float32=True),
    dict(type='PhotoMetricDistortionMultiViewImage'),
    dict(type='ResizeMultiViewImages',
         size=img_size, # H, W
         change_intrinsics=True,
         ),
    dict(type='Normalize3D', **img_norm_cfg),
    dict(type='PadMultiViewImages', size_divisor=32),
    dict(type='FormatBundleMap'),
    dict(type='Collect3D', keys=['img', 'vectors', 'semantic_mask'], meta_keys=(
        'token', 'ego2img', 'sample_idx', 'ego2global_translation',
        'ego2global_rotation', 'img_shape', 'scene_name'))
]

# data processing pipelines
test_pipeline = [
    dict(type='LoadMultiViewImagesFromFiles', to_float32=True),
    dict(type='ResizeMultiViewImages',
         size=img_size, # H, W
         change_intrinsics=True,
         ),
    dict(type='Normalize3D', **img_norm_cfg),
    dict(type='PadMultiViewImages', size_divisor=32),
    dict(type='FormatBundleMap'),
    dict(type='Collect3D', keys=['img'], meta_keys=(
        'token', 'ego2img', 'sample_idx', 'ego2global_translation',
        'ego2global_rotation', 'img_shape', 'scene_name'))
]

# configs for evaluation code
# DO NOT CHANGE
eval_config = dict(
    type='NuscDataset',
    data_root='./datasets/nuscenes',
    ann_file='./datasets/nuscenes/nuscenes_map_infos_val.pkl',
    meta=meta,
    roi_size=roi_size,
    cat2id=cat2id,
    pipeline=[
        dict(
            type='VectorizeMap',
            coords_dim=coords_dim,
            simplify=True,
            normalize=False,
            roi_size=roi_size
        ),
        dict(type='FormatBundleMap'),
        dict(type='Collect3D', keys=['vectors',], meta_keys=['token', 'ego2img', 'sample_idx', 'ego2global_translation',
        'ego2global_rotation', 'img_shape', 'scene_name'])
    ],
    interval=1,
)


match_config = dict(
    type='NuscDataset',
    data_root='./datasets/nuscenes',
    ann_file='./datasets/nuscenes/nuscenes_map_infos_val.pkl',
    meta=meta,
    roi_size=roi_size,
    cat2id=cat2id,
    pipeline=[
        dict(
            type='VectorizeMap',
            coords_dim=coords_dim,
            simplify=False,
            normalize=True,
            roi_size=roi_size,
            sample_num=num_points,
        ),
        dict(
            type='RasterizeMap',   
            roi_size=roi_size,
            coords_dim=coords_dim,
            canvas_size=canvas_size,
            thickness=thickness,
        ),
        dict(type='FormatBundleMap'),
        dict(type='Collect3D', keys=['vectors', 'semantic_mask'], meta_keys=['token', 'ego2img', 'sample_idx', 'ego2global_translation',
        'ego2global_rotation', 'img_shape', 'scene_name'])
    ],
    interval=1,
)

# dataset configs
data = dict(
    samples_per_gpu=batch_size,
    workers_per_gpu=8,
    train=dict(
        type='NuscDataset',
        data_root='./datasets/nuscenes',
        ann_file='./datasets/nuscenes/nuscenes_map_infos_train.pkl',
        meta=meta,
        roi_size=roi_size,
        cat2id=cat2id,
        pipeline=train_pipeline,
        seq_split_num=-2,
        matching=True,
        multi_frame=5,
        sampling_span=10,
    ),
    val=dict(
        type='NuscDataset',
        data_root='./datasets/nuscenes',
        ann_file='./datasets/nuscenes/nuscenes_map_infos_val.pkl',
        meta=meta,
        roi_size=roi_size,
        cat2id=cat2id,
        pipeline=test_pipeline,
        eval_config=eval_config,
        test_mode=True,
        seq_split_num=1,
    ),
    test=dict(
        type='NuscDataset',
        data_root='./datasets/nuscenes',
        ann_file='./datasets/nuscenes/nuscenes_map_infos_val.pkl',
        meta=meta,
        roi_size=roi_size,
        cat2id=cat2id,
        pipeline=test_pipeline,
        eval_config=eval_config,
        test_mode=True,
        seq_split_num=1,
    ),
    shuffler_sampler=dict(type='DistributedGroupSampler'),
    nonshuffler_sampler=dict(type='DistributedSampler')
)

train_dataloader = dict(
    batch_size=batch_size,
    num_workers=data['workers_per_gpu'],
    persistent_workers=True,
    sampler=data['shuffler_sampler'],
    dataset=dict(
        type='NuscDataset',
        data_root='./datasets/nuscenes',
        ann_file='./datasets/nuscenes/nuscenes_map_infos_train.pkl',
        meta=meta,
        roi_size=roi_size,
        cat2id=cat2id,
        pipeline=train_pipeline,
        seq_split_num=-2,
        matching=True,
        multi_frame=5,
        sampling_span=10,
    ))

val_dataloader_cfg = dict(
    batch_size=1,
    num_workers=data['workers_per_gpu'],
    persistent_workers=True,
    sampler=data['nonshuffler_sampler'],
    dataset=dict(
        type='NuscDataset',
        data_root='./datasets/nuscenes',
        ann_file='./datasets/nuscenes/nuscenes_map_infos_val.pkl',
        meta=meta,
        roi_size=roi_size,
        cat2id=cat2id,
        pipeline=test_pipeline,
        eval_config=eval_config,
        test_mode=True,
        seq_split_num=1,
    ))

timestamp = (__import__('datetime')
             .datetime
             .now()
             .strftime('%Y_%m_%d_%H_%M_%S'))

evaluation = dict(
    interval=num_epochs_interval*num_iters_per_epoch,
    by_epoch = False,
    jsonfile_prefix='val/${work_dir}/' + timestamp)

custom_hooks = [
    dict(
        type='CustomDistEvalHook',
        dataloader=val_dataloader_cfg,
        priority='LOW',
        **evaluation
    )
]

# ---- Optimizer overrides (MMEngine-style) ----
optim_wrapper = dict(
    paramwise_cfg=dict(
        custom_keys={
            'backbone.img_backbone': dict(lr_mult=0.1),
            'backbone.img_neck': dict(lr_mult=0.5),
            'backbone.transformer': dict(lr_mult=0.5),
            'backbone.positional_encoding': dict(lr_mult=0.5),
            'seg_decoder': dict(lr_mult=0.5),
        }),
    clip_grad=dict(max_norm=35, norm_type=2),
)

# learning rate schedulers (MMEngine style)
param_scheduler = [
    # Linear warm-up
    dict(
        type='LinearLR',
        start_factor=1.0/3,
        by_epoch=False,
        begin=0,
        end=500),
    # Cosine annealing after warm-up
    dict(
        type='CosineAnnealingLR',
        T_max=total_iters,
        eta_min=5e-4 * 3e-3,
        by_epoch=False,
        begin=500,
        end=total_iters)
]

#evaluation = dict(interval=1) # for debugging use..
find_unused_parameters = True #### when use checkpoint, find_unused_parameters must be False

# default hooks overrides
default_hooks = dict(
    logger=dict(type='LoggerHook', interval=50),
    checkpoint=dict(type='CheckpointHook', by_epoch=False, interval=num_epochs_interval*num_iters_per_epoch)
)

train_cfg=dict(
    by_epoch=False, 
    max_iters=num_epochs * num_iters_per_epoch
)

runner_type = 'MyRunnerWrapper' 

SyncBN = True

load_from = "work_dirs/maptracker_nusc_oldsplit_5frame_span10_stage2_warmup/epoch_1.pth"
