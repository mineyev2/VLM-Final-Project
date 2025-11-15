"""
SST Wrapper - Uses mmdet3d's built-in components
"""
import sys
from pathlib import Path

import torch
import torch.nn as nn
from mmengine.config import Config
from mmengine.registry import init_default_scope
from mmengine.runner import load_checkpoint

from mmengine import init_default_scope


project_root = Path(__file__).parent.parent.parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))
    
# ✅ Use mmdet3d's built-in registry and classes
from mmdet3d.registry import MODELS as MMDET3D_MODELS
init_default_scope('mmdet3d')

def build_sst(sst_config):
    """Build SST model from config using mmdet3d's components."""
    
    # Initialize mmdet3d scope
    init_default_scope('mmdet3d')
    
    # Load config
    cfg = Config.fromfile(sst_config)
    
    print(f"[SST] Building model from config: {sst_config}")
    
    # Build model using mmdet3d's registry
    # This will use mmdet3d's built-in DynamicVFE, SSTInputLayerV2, SSTv2, etc.
    model = MMDET3D_MODELS.build(cfg.model)
    
    return model

@MMDET3D_MODELS.register_module()
class LidarEncoderSST(nn.Module):
    """SST-based LiDAR encoder wrapper."""
    
    def __init__(self, sst_config, clip_embedding_dim=512, checkpoint=None):
        super().__init__()
        
        print(f"\n{'='*70}")
        print(f"[LidarEncoderSST] Initializing")
        print(f"{'='*70}")
        print(f"Config: {sst_config}")
        print(f"Output dim: {clip_embedding_dim}")
        
        # Build SST model using mmdet3d's components
        self._sst = build_sst(sst_config)
        
        # Extract backbone output shape
        cfg = Config.fromfile(sst_config)
        backbone_output_shape = cfg.model.backbone.get('output_shape', [80, 80])
        backbone_channels = cfg.model.backbone.get('conv_out_channel', 128)
        
        print(f"\n[LidarEncoderSST] Building attention pooler...")
        print(f"  Input: [{backbone_channels}, {backbone_output_shape[0]}, {backbone_output_shape[1]}]")
        print(f"  Output: {clip_embedding_dim}")
        
        # Attention pooler (you still need this)
        from src.models.attention_pool import AttentionPool2d
        self._pooler = AttentionPool2d(
            spacial_dim=backbone_output_shape[0],
            embed_dim=clip_embedding_dim,
            num_heads=8,
            input_dim=backbone_channels,
        )
        
        # Load checkpoint if provided
        if checkpoint:
            print(f"\n[LidarEncoderSST] Loading checkpoint: {checkpoint}")
            load_checkpoint(self, checkpoint, map_location='cpu', strict=False)
        
        print(f"\n{'='*70}")
        print("[LidarEncoderSST] ✓ Initialization complete!")
        print(f"{'='*70}\n")
    
    def forward(self, point_clouds, no_pooling=False, return_attention=False):
        """Forward pass through SST encoder.
        
        Args:
            point_clouds (list[Tensor]): List of point clouds, each [N, 4]
            no_pooling (bool): If True, return BEV features without pooling
            return_attention (bool): If True, also return attention weights
            
        Returns:
            Tensor or tuple: Pooled features [B, C] or (features, attention)
        """
        # Extract BEV features using SST
        # This calls: voxel_encoder → middle_encoder → backbone
        bev_features = self._sst.extract_feat(point_clouds)
        
        # bev_features is typically a tuple, take first element
        if isinstance(bev_features, (list, tuple)):
            bev_features = bev_features[0]
        
        # Return spatial features if no pooling requested
        if no_pooling:
            return bev_features
        
        # Apply attention pooling
        pooled_features, attention_weights = self._pooler(
            bev_features,
            no_pooling=False,
            return_attention=return_attention
        )
        
        if return_attention:
            return pooled_features, attention_weights
        else:
            return pooled_features
