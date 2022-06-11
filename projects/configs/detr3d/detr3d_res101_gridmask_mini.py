_base_ = [
    './detr3d_res101_gridmask.py',
]


# data_root = '/home/master/10/cytseng/mmdetection3d/data/nuscenes/'
# data_root = '/work/sty61010/datasets/nuscenes/v1.0-mini/'
# data_root = '/work/sty61010/datasets/nuscenes/v1.0-trainval/'

data_root = '/home/master/10/cytseng/data/sets/nuscenes/v1.0-mini/'
# data_root = '/home/master/10/cytseng/data/sets/nuscenes/v1.0-trainval/'

db_sampler = dict(
    data_root=data_root,
    info_path=data_root + 'nuscenes_dbinfos_train.pkl',
    )


data_length = 6000
data = dict(
    samples_per_gpu=1,
    workers_per_gpu=4,
    train=dict(
        data_root=data_root,
        ann_file=data_root + 'nuscenes_infos_train.pkl',
        data_length=data_length,
    ),
    val=dict(
        data_root=data_root,
        ann_file=data_root + 'nuscenes_infos_val.pkl',
        ),
    test=dict(
        data_root=data_root,
        ann_file=data_root + 'nuscenes_infos_val.pkl',
        ))