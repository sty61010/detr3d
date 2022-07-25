_base_ = [
    './deformable_decoder_detr3d_res50dcn_gridmask_l3_512_1408.py',
]
data_length = 6000
data = dict(
    train=dict(
        data_length=data_length,
    ),
)

# If point cloud range is changed, the models should also change their point
# cloud range accordingly
point_cloud_range = [-51.2, -51.2, -5.0, 51.2, 51.2, 3.0]
voxel_size = [0.2, 0.2, 8]
embed_dims = 256
num_levels = 2
# num_levels = 4
# grid_size=[2.048, 2.048, 8]
grid_size = [1.024, 1.024, 8]

model = dict(
    type='Detr3D',
    use_grid_mask=True,
    img_backbone=dict(
        type='ResNet',
        depth=50,
        num_stages=4,
        out_indices=(2, 3,),
        # out_indices=(0, 1, 2, 3),
        # frozen_stages=1,
        frozen_stages=-1,
        norm_cfg=dict(type='BN2d', requires_grad=False),
        norm_eval=True,
        style='caffe',
        dcn=dict(type='DCNv2', deform_groups=1, fallback_on_stride=False),
        stage_with_dcn=(False, False, True, True),
        with_cp=True,
        pretrained='ckpts/resnet50_msra-5891d200.pth',
    ),
    # img_neck=dict(
    #     type='FPN',
    #     in_channels=[512, 1024, 2048],
    #     # in_channels=[256, 512, 1024, 2048],
    #     out_channels=embed_dims,
    #     # start_level=1,
    #     start_level=0,
    #     add_extra_convs='on_output',
    #     num_outs=num_levels,
    #     relu_before_extra_convs=True,
    # ),
    img_neck=dict(
        type='CPFPN',
        in_channels=[1024, 2048],
        out_channels=embed_dims,
        num_outs=num_levels,
    ),
    pts_bbox_head=dict(
        type='DeformableDetr3DHead',
        num_query=900,
        num_classes=10,
        in_channels=embed_dims,
        sync_cls_avg_factor=True,
        with_box_refine=True,
        as_two_stage=False,

        transformer=dict(
            type='DeformableDetr3DTransformer',
            grid_size=grid_size,
            pc_range=point_cloud_range,

            decoder=dict(
                type='DeformableDetr3DTransformerDecoder',
                num_layers=6,
                return_intermediate=True,
                transformerlayers=dict(
                    # type='BaseTransformerLayer',
                    type='DepthCrossDecoderLayer',
                    attn_cfgs=[
                        dict(
                            type='MultiheadAttention',
                            embed_dims=embed_dims,
                            num_heads=8,
                            dropout=0.1,
                        ),
                        dict(
                            type='DeformableCrossAttention',
                            attn_cfg=dict(
                                type='MultiScaleDeformableAttention',
                                num_levels=num_levels,
                                # embed_dims=256,
                            ),
                            pc_range=point_cloud_range,
                            num_points=1,
                            embed_dims=embed_dims
                        ),

                        # dict(
                        #     type='MultiheadAttention',
                        #     embed_dims=embed_dims,
                        #     num_heads=8,
                        #     dropout=0.1,
                        # ),
                    ],
                    feedforward_channels=512,
                    ffn_dropout=0.1,
                    operation_order=(
                        'self_attn', 'norm',
                        'cross_view_attn', 'norm',
                        # 'cross_depth_attn', 'norm',
                        'ffn', 'norm',
                    )
                )
            )
        ),
        bbox_coder=dict(
            type='NMSFreeCoder',
            post_center_range=[-61.2, -61.2, -10.0, 61.2, 61.2, 10.0],
            pc_range=point_cloud_range,
            max_num=300,
            voxel_size=voxel_size,
            num_classes=10),
        loss_cls=dict(
            type='FocalLoss',
            use_sigmoid=True,
            gamma=2.0,
            alpha=0.25,
            loss_weight=2.0),
        loss_bbox=dict(type='L1Loss', loss_weight=0.25),
        loss_iou=dict(type='GIoULoss', loss_weight=0.0)),
    # model training and testing settings
    train_cfg=dict(pts=dict(
        grid_size=[512, 512, 1],
        voxel_size=voxel_size,
        point_cloud_range=point_cloud_range,
        out_size_factor=4,
        assigner=dict(
            type='HungarianAssigner3D',
            cls_cost=dict(type='FocalLossCost', weight=2.0),
            reg_cost=dict(type='BBox3DL1Cost', weight=0.25),
            iou_cost=dict(type='IoUCost', weight=0.0),  # Fake cost. This is just to make it compatible with DETR head.
            pc_range=point_cloud_range
        ))
    )
)
