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


def build_sst(sst_config, encoder_only=True):
    """Build SST model from config using mmdet3d's components.
    
    Args:
        sst_config (str): Path to config file
        encoder_only (bool): If True, build only encoder components (safer).
                            If False, build full detector (may need bbox_head).
    
    Returns:
        nn.Module: SST model or encoder-only wrapper
    """
    # Initialize mmdet3d scope
    init_default_scope('mmdet3d')
    
    # Load config
    cfg = Config.fromfile(sst_config)
    
    print(f"[SST] Building model from config: {sst_config}")
    print(f"[SST] Mode: {'Encoder-only' if encoder_only else 'Full detector'}")
    
    if encoder_only:
        # Build individual encoder components (more robust, avoids bbox_head issues)
        print("[SST] Building encoder components individually...")
        
        voxel_encoder = MMDET3D_MODELS.build(cfg.model.voxel_encoder)
        print(f"[SST]   ✓ Voxel encoder: {cfg.model.voxel_encoder.type}")
        
        middle_encoder = MMDET3D_MODELS.build(cfg.model.middle_encoder)
        print(f"[SST]   ✓ Middle encoder: {cfg.model.middle_encoder.type}")
        
        backbone = MMDET3D_MODELS.build(cfg.model.backbone)
        print(f"[SST]   ✓ Backbone: {cfg.model.backbone.type}")
        
        # Wrap in a simple container that mimics detector interface
        class SSTEncoderOnly(nn.Module):
            """Encoder-only wrapper that provides extract_feat() interface."""
            
            def __init__(self, voxel_encoder, middle_encoder, backbone):
                super().__init__()
                self.voxel_encoder = voxel_encoder
                self.middle_encoder = middle_encoder
                self.backbone = backbone
            
            def extract_feat(self, points):
                """Extract features from point cloud.
                
                Args:
                    points: Point cloud data
                    
                Returns:
                    Tensor: Backbone features
                """
                # 1. Voxelize and encode features
                voxel_features = self.voxel_encoder(points)
                
                # 2. Middle encoder (window preparation for SST)
                encoder_features = self.middle_encoder(voxel_features)
                
                # 3. SST backbone (transformer layers)
                backbone_features = self.backbone(encoder_features)
                
                return backbone_features
        
        model = SSTEncoderOnly(voxel_encoder, middle_encoder, backbone)
        print("[SST] ✓ Built encoder-only model")
        
    else:
        # Build full detector model (requires bbox_head in config)
        print("[SST] Building full detector model...")
        model = MMDET3D_MODELS.build(cfg.model)
        print(f"[SST] ✓ Built full detector: {cfg.model.type}")
    
    return model


@MMDET3D_MODELS.register_module()
class LidarEncoderSST(nn.Module):
    """SST-based LiDAR encoder wrapper."""
    
    def __init__(self, sst_config, clip_embedding_dim=512, checkpoint=None, 
                 encoder_only=True):
        """Initialize LidarEncoderSST.
        
        Args:
            sst_config (str): Path to SST config file
            clip_embedding_dim (int): Output embedding dimension for CLIP
            checkpoint (str, optional): Path to pretrained checkpoint
            encoder_only (bool): If True, build only encoder (recommended).
                                If False, build full detector (needs bbox_head).
        """
        super().__init__()
        
        print(f"\n{'='*70}")
        print(f"[LidarEncoderSST] Initializing")
        print(f"{'='*70}")
        print(f"Config: {sst_config}")
        print(f"Output dim: {clip_embedding_dim}")
        print(f"Mode: {'Encoder-only' if encoder_only else 'Full detector'}")
        
        # Build SST model using mmdet3d's components
        self._sst = build_sst(sst_config, encoder_only=encoder_only)
        
        # Extract backbone output shape from config
        cfg = Config.fromfile(sst_config)
        backbone_output_shape = cfg.model.backbone.get('output_shape', [80, 80])
        backbone_channels = cfg.model.backbone.get('conv_out_channel', 128)
        
        print(f"\n[LidarEncoderSST] Building attention pooler...")
        print(f"  Input: [{backbone_channels}, {backbone_output_shape[0]}, {backbone_output_shape[1]}]")
        print(f"  Output: {clip_embedding_dim}")
        
        # Attention pooler to convert BEV features to CLIP embedding
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
            print("[LidarEncoderSST] ✓ Checkpoint loaded")
        
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
            Tensor or tuple: 
                - If no_pooling=True: BEV features [B, C, H, W]
                - If return_attention=True: (pooled_features, attention_weights)
                - Otherwise: Pooled features [B, clip_embedding_dim]
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
        
        # Apply attention pooling to get fixed-size CLIP embedding
        pooled_features, attention_weights = self._pooler(
            bev_features,
            no_pooling=False,
            return_attention=return_attention
        )
        
        if return_attention:
            return pooled_features, attention_weights
        else:
            return pooled_features
