"""
SST Encoder-Only Configuration
For LidarCLIP with Qwen2.5-VL
Compatible with mmcv 2.1.0, mmdet3d 1.4.0
"""
custom_imports = dict(
    imports=[
        'src.models.sst_v2',           # SSTv2
        'src.models.sst_input_layer_v2'       # SSTInputLayerV2
    ],
    allow_failed_imports=False
)

# Point cloud range and voxel size
voxel_size = [0.5, 0.5, 6]
point_cloud_range = [0, -20.0, -2, 40.0, 20.0, 4]
window_shape = [12, 12, 1]
sparse_shape = [80, 80, 1]

# Drop info
drop_info_training = {
    0: {'max_tokens': 30, 'drop_range': (0, 30)},
    1: {'max_tokens': 60, 'drop_range': (30, 60)},
    2: {'max_tokens': 100, 'drop_range': (60, 100)},
    3: {'max_tokens': 200, 'drop_range': (100, 200)},
    4: {'max_tokens': 250, 'drop_range': (200, 100000)},
}
drop_info_test = {
    0: {'max_tokens': 30, 'drop_range': (0, 30)},
    1: {'max_tokens': 60, 'drop_range': (30, 60)},
    2: {'max_tokens': 100, 'drop_range': (60, 100)},
    3: {'max_tokens': 200, 'drop_range': (100, 200)},
    4: {'max_tokens': 256, 'drop_range': (200, 100000)},
}
drop_info = (drop_info_training, drop_info_test)
num_encoder_layers = 4

# Direct component configs (no full detector wrapper)
model = dict(
    voxel_encoder=dict(
        type='DynamicVFE',
        in_channels=4,
        feat_channels=[64, 128],
        with_distance=False,
        voxel_size=voxel_size,
        with_cluster_center=True,
        with_voxel_center=True,
        point_cloud_range=point_cloud_range,
        norm_cfg=dict(type='BN1d', eps=1e-3, momentum=0.01)
    ),
    
    middle_encoder=dict(
        type='SSTInputLayerV2',
        window_shape=window_shape,
        sparse_shape=sparse_shape,
        shuffle_voxels=True,
        debug=True,
        drop_info=drop_info,
        pos_temperature=10000,
        normalize_pos=False,
    ),
    
    backbone=dict(
        type='SSTv2',
        d_model=[128] * num_encoder_layers,
        nhead=[8] * num_encoder_layers,
        num_blocks=num_encoder_layers,
        dim_feedforward=[256] * num_encoder_layers,
        dropout=0.0,
        activation='gelu',
        output_shape=[80, 80],
        num_attached_conv=0,
        conv_in_channel=128,
        conv_out_channel=128,
        norm_cfg=dict(type='BN2d', eps=1e-3, momentum=0.01),
        conv_cfg=dict(type='Conv2d', bias=False),
        debug=True,
        checkpoint_blocks=[],
        layer_cfg=dict(),
    ),
)
