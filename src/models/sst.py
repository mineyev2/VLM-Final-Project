import sys
import os
from mmengine.config import Config
from mmengine.registry import init_default_scope
from mmdet3d.registry import MODELS as MMDET3D_MODELS

import torch
from torch import nn

from .attention_pool import AttentionPool2d
from .sst_encoder_only_config import model as sst_model_conf

# Add the cloned lidarclip repo to sys.path
lidarclip_sst_path = os.path.join(os.path.dirname(__file__), "../../lidarclip_repo/SST")
sys.path.insert(0, lidarclip_sst_path)

# Import SSTv2 from the cloned repo
from mmdet3d.models.backbones.sst_v2 import SSTv2

# Register SSTv2 if not already registered
if 'SSTv2' not in MMDET3D_MODELS.module_dict:
    MMDET3D_MODELS.register_module()(SSTv2)

class LidarEncoderSST(nn.Module):
    def __init__(self, sst_config, clip_embedding_dim=512, checkpoint=None):
        super().__init__()
        init_default_scope('mmdet3d')
        
        # Load config
        cfg = Config.fromfile(sst_config)
        m = cfg.model
        
        # v1 -> v2 compatibility for voxel_layer
        if 'voxel_layer' in m:
            legacy_voxel = m.pop('voxel_layer')
            dp = m.setdefault('data_preprocessor', dict(type='Det3DDataPreprocessor'))
            vc = dp.setdefault('voxelize_cfg', {})
            if isinstance(legacy_voxel, dict):
                vc.update(legacy_voxel)
        
        # Extract backbone if full detector
        if m.get('type') in ['DynamicVoxelNet', 'VoxelNet'] and 'backbone' in m:
            backbone_cfg = m['backbone']
            self._sst = MMDET3D_MODELS.build(backbone_cfg)
        else:
            self._sst = MMDET3D_MODELS.build(m)
        
        if hasattr(self._sst, 'init_weights'):
            self._sst.init_weights()
        
        # Load checkpoint if provided
        if checkpoint:
            from mmengine.runner import load_checkpoint
            load_checkpoint(self._sst, checkpoint, map_location='cpu')
        
        self._pooler = AttentionPool2d(
            spacial_dim=sst_model_conf["backbone"]["output_shape"][0],
            embed_dim=clip_embedding_dim,
            num_heads=8,
            input_dim=sst_model_conf["backbone"]["conv_out_channel"],
        )

    def extract_lidar_feat(self, point_cloud, img_metas=None):
        feat = self._sst.extract_feat(point_cloud, img_metas)
        if isinstance(feat, (list, tuple)):
            return feat[0]
        return feat

    def forward(self, point_cloud, no_pooling=False, return_attention=False):
        lidar_features = self.extract_lidar_feat(point_cloud)
        pooled_feature, attn_weights = self._pooler(lidar_features, no_pooling, return_attention)
        if return_attention:
            return pooled_feature, attn_weights
        return pooled_feature