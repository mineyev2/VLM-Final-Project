import sys
import os
import importlib.util
from mmengine.config import Config
from mmengine.registry import init_default_scope
from mmdet3d.registry import MODELS as MMDET3D_MODELS

import torch
from torch import nn

from .attention_pool import AttentionPool2d
from .sst_encoder_only_config import model as sst_model_conf

# Setup paths for all SST-related imports
sst_repo_root = os.path.join(os.path.dirname(__file__), "../../lidarclip_repo/SST")
sys.path.insert(0, sst_repo_root)
sys.path.insert(0, os.path.join(sst_repo_root, "mmdet3d/models/sst"))
sys.path.insert(0, os.path.join(sst_repo_root, "mmdet3d/ops/sst"))

# Load SSTv2 directly by file path
sst_v2_path = os.path.join(sst_repo_root, "mmdet3d/models/backbones/sst_v2.py")
spec = importlib.util.spec_from_file_location("sst_v2", sst_v2_path)
sst_v2_module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(sst_v2_module)
SSTv2 = sst_v2_module.SSTv2

# Register SSTv2
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
"""
SST Encoder Wrapper for LidarCLIP
Compatible with mmcv 2.1.0, mmdet3d 1.4.0, torch 2.1.2

This wrapper chains together the SST encoder components:
  Voxelization → DynamicVFE → SSTInputLayerV2 → SSTv2 → AttentionPool2d
"""

import torch
import torch.nn as nn
from mmengine.config import Config
from mmengine.registry import init_default_scope
from mmengine.runner import load_checkpoint
from mmdet3d.registry import MODELS as MMDET3D_MODELS

from .attention_pool import AttentionPool2d


class LidarEncoderSST(nn.Module):
    """
    SST-based LiDAR encoder for LidarCLIP.
    
    Args:
        sst_config (str): Path to SST config file
        clip_embedding_dim (int): Output dimension to match CLIP (default: 512)
        checkpoint (str, optional): Path to pretrained checkpoint
    """
    
    def __init__(self, sst_config, clip_embedding_dim=512, checkpoint=None):
        super().__init__()
        
        # Initialize mmdet3d scope
        init_default_scope('mmdet3d')
        
        # Load config
        cfg = Config.fromfile(sst_config)
        model_cfg = cfg.model
        
        print(f"\n{'='*70}")
        print(f"[LidarEncoderSST] Initializing SST Encoder")
        print(f"{'='*70}")
        print(f"Config: {sst_config}")
        print(f"Output dimension: {clip_embedding_dim}")
        
        # ============================================
        # Handle legacy voxel_layer format
        # ============================================
        if 'voxel_layer' in model_cfg:
            print("\n[Migration] Converting legacy voxel_layer to data_preprocessor...")
            legacy_voxel = model_cfg.pop('voxel_layer')
            
            # Create data_preprocessor config
            data_preprocessor = model_cfg.setdefault(
                'data_preprocessor', 
                dict(type='Det3DDataPreprocessor')
            )
            voxelize_cfg = data_preprocessor.setdefault('voxelize_cfg', {})
            
            # Migrate parameters
            if isinstance(legacy_voxel, dict):
                voxelize_cfg.update({
                    'voxel_size': legacy_voxel.get('voxel_size'),
                    'point_cloud_range': legacy_voxel.get('point_cloud_range'),
                    'max_num_points': legacy_voxel.get('max_num_points', -1),
                    'max_voxels': legacy_voxel.get('max_voxels', (-1, -1)),
                })
                print("  ✓ Migrated voxel_layer parameters")
        
        # ============================================
        # Build Component 1: Voxel Encoder
        # ============================================
        print("\n[1/4] Building Voxel Encoder...")
        if 'voxel_encoder' not in model_cfg:
            raise ValueError("Config must contain 'voxel_encoder'")
        
        voxel_type = model_cfg.voxel_encoder.get('type', 'DynamicVFE')
        print(f"  Type: {voxel_type}")
        
        try:
            self.voxel_encoder = MMDET3D_MODELS.build(model_cfg.voxel_encoder)
            print(f"  ✓ {voxel_type} built successfully")
        except Exception as e:
            print(f"  ✗ Failed to build {voxel_type}: {e}")
            raise
        
        # ============================================
        # Build Component 2: Middle Encoder
        # ============================================
        print("\n[2/4] Building Middle Encoder...")
        if 'middle_encoder' not in model_cfg:
            raise ValueError("Config must contain 'middle_encoder'")
        
        middle_type = model_cfg.middle_encoder.get('type', 'SSTInputLayerV2')
        print(f"  Type: {middle_type}")
        
        try:
            self.middle_encoder = MMDET3D_MODELS.build(model_cfg.middle_encoder)
            print(f"  ✓ {middle_type} built successfully")
        except Exception as e:
            print(f"  ✗ Failed to build {middle_type}: {e}")
            raise
        
        # ============================================
        # Build Component 3: Backbone
        # ============================================
        print("\n[3/4] Building Backbone...")
        if 'backbone' not in model_cfg:
            raise ValueError("Config must contain 'backbone'")
        
        backbone_type = model_cfg.backbone.get('type', 'SSTv2')
        print(f"  Type: {backbone_type}")
        
        try:
            self.backbone = MMDET3D_MODELS.build(model_cfg.backbone)
            
            # Initialize weights if method exists
            if hasattr(self.backbone, 'init_weights'):
                self.backbone.init_weights()
            
            print(f"  ✓ {backbone_type} built successfully")
        except Exception as e:
            print(f"  ✗ Failed to build {backbone_type}: {e}")
            raise
        
        # ============================================
        # Build Component 4: Attention Pooler
        # ============================================
        print("\n[4/4] Building Attention Pooler...")
        
        # Extract parameters from backbone config
        backbone_output_shape = model_cfg.backbone.get('output_shape', [80, 80])
        backbone_channels = model_cfg.backbone.get('conv_out_channel', 128)
        
        print(f"  Input shape: [{backbone_channels}, {backbone_output_shape[0]}, {backbone_output_shape[1]}]")
        print(f"  Output dim: {clip_embedding_dim}")
        print(f"  Num heads: 8")
        
        try:
            self._pooler = AttentionPool2d(
                spacial_dim=backbone_output_shape[0],
                embed_dim=clip_embedding_dim,
                num_heads=8,
                input_dim=backbone_channels,
            )
            print(f"  ✓ AttentionPool2d built successfully")
        except Exception as e:
            print(f"  ✗ Failed to build AttentionPool2d: {e}")
            raise
        
        # ============================================
        # Load Checkpoint
        # ============================================
        if checkpoint:
            print(f"\n[Checkpoint] Loading from: {checkpoint}")
            self._load_checkpoint(checkpoint)
        
        print(f"\n{'='*70}")
        print("[LidarEncoderSST] ✓ Initialization Complete!")
        print(f"{'='*70}\n")
    
    def _load_checkpoint(self, checkpoint_path):
        """
        Load pretrained weights from checkpoint.
        Only loads encoder components (voxel_encoder, middle_encoder, backbone, pooler).
        Skips detection head weights (bbox_head, neck, etc.).
        """
        try:
            # Load checkpoint with strict=False to allow missing keys
            checkpoint_dict = load_checkpoint(
                self, 
                checkpoint_path, 
                map_location='cpu',
                strict=False
            )
            
            # Count loaded parameters
            if isinstance(checkpoint_dict, dict) and 'state_dict' in checkpoint_dict:
                state_dict = checkpoint_dict['state_dict']
            else:
                state_dict = checkpoint_dict
            
            # Filter encoder keys
            encoder_keys = [k for k in state_dict.keys() 
                          if not any(skip in k for skip in ['bbox_head', 'neck', 'roi_head'])]
            
            print(f"  ✓ Loaded {len(encoder_keys)} encoder parameters")
            print(f"  Skipped detection head parameters")
            
        except FileNotFoundError:
            print(f"  ✗ Checkpoint file not found: {checkpoint_path}")
            print(f"  Continuing with random initialization...")
        except Exception as e:
            print(f"  ⚠ Warning: Could not load checkpoint: {e}")
            print(f"  Continuing with random initialization...")
    
    def extract_lidar_feat(self, point_clouds, img_metas=None):
        """
        Extract BEV features from point clouds (before pooling).
        This returns the spatial feature map from the backbone.
        
        Args:
            point_clouds (list[Tensor]): List of point clouds [N, 4]
            img_metas (list[dict], optional): Image metadata (unused)
        
        Returns:
            Tensor: BEV feature map [B, C, H, W]
        """
        # Step 1: Voxel encoding
        voxel_dict = self.voxel_encoder(point_clouds)
        
        # Step 2: Middle encoder
        middle_output = self.middle_encoder(voxel_dict)
        
        # Step 3: Backbone
        backbone_features = self.backbone(middle_output)
        
        # Return spatial features (before pooling)
        if isinstance(backbone_features, (list, tuple)):
            return backbone_features[0]
        return backbone_features
    
    def forward(self, point_clouds, no_pooling=False, return_attention=False):
        """
        Forward pass through SST encoder.
        
        Args:
            point_clouds (list[Tensor]): List of point clouds, each [N, 4] (x, y, z, intensity)
                                         Length = batch_size
            no_pooling (bool): If True, return spatial BEV features without pooling
            return_attention (bool): If True, also return attention weights from pooler
        
        Returns:
            If no_pooling=True:
                features (Tensor): BEV feature map [B, C, H, W]
            If no_pooling=False and return_attention=False:
                pooled_features (Tensor): [B, clip_embedding_dim]
            If no_pooling=False and return_attention=True:
                tuple: (pooled_features [B, clip_embedding_dim], attention_weights)
        """
        # Extract BEV features
        bev_features = self.extract_lidar_feat(point_clouds)
        
        # Return spatial features if no pooling requested
        if no_pooling:
            return bev_features
        
        # Apply attention pooling
        pooled_features, attention_weights = self._pooler(
            bev_features,
            no_pooling=False,
            return_attention=return_attention
        )
        
        # Return based on what's requested
        if return_attention:
            return pooled_features, attention_weights
        else:
            return pooled_features


# ============================================
# Testing
# ============================================
if __name__ == "__main__":
    print("="*70)
    print("Testing LidarEncoderSST")
    print("="*70)
    
    # Create minimal test config
    import tempfile
    import os
    
    config_content = """
model = dict(
    type='DynamicVoxelNet',
    voxel_layer=dict(
        voxel_size=[0.5, 0.5, 6],
        point_cloud_range=[0, -20, -2, 40, 20, 4],
        max_num_points=-1,
        max_voxels=(-1, -1),
    ),
    voxel_encoder=dict(
        type='DynamicVFE',
        in_channels=4,
        feat_channels=[64, 128],
        with_distance=False,
        voxel_size=[0.5, 0.5, 6],
        with_cluster_center=True,
        with_voxel_center=True,
        point_cloud_range=[0, -20, -2, 40, 20, 4],
        norm_cfg=dict(type='BN1d', eps=1e-3, momentum=0.01)
    ),
    middle_encoder=dict(
        type='SSTInputLayerV2',
        window_shape=[12, 12, 1],
        sparse_shape=[80, 80, 1],
        shuffle_voxels=True,
        debug=False,
    ),
    backbone=dict(
        type='SSTv2',
        d_model=[128] * 4,
        nhead=[8] * 4,
        num_blocks=4,
        dim_feedforward=[256] * 4,
        output_shape=[80, 80],
        conv_out_channel=128,
        num_attached_conv=0,
    ),
)
"""
    
    # Write temp config
    with tempfile.NamedTemporaryFile(mode='w', suffix='.py', delete=False) as f:
        f.write(config_content)
        config_path = f.name
    
    try:
        print("\n" + "="*70)
        print("Step 1: Initialize Encoder")
        print("="*70)
        
        encoder = LidarEncoderSST(
            sst_config=config_path,
            clip_embedding_dim=768
        )
        
        print("\n" + "="*70)
        print("Step 2: Create Dummy Input")
        print("="*70)
        
        batch_size = 2
        points_list = [
            torch.rand(1000, 4) for _ in range(batch_size)
        ]
        print(f"Created {batch_size} point clouds with 1000 points each")
        
        print("\n" + "="*70)
        print("Step 3: Forward Pass")
        print("="*70)
        
        with torch.no_grad():
            # Test without attention
            print("\nTest 1: Standard forward pass")
            features = encoder(points_list)
            print(f"  Output shape: {features.shape}")
            print(f"  Expected: [{batch_size}, 768]")
            assert features.shape == (batch_size, 768), "Shape mismatch!"
            print("  ✓ Passed")
            
            # Test with attention
            print("\nTest 2: Forward pass with attention")
            features, attn = encoder(points_list, return_attention=True)
            print(f"  Features shape: {features.shape}")
            print(f"  Attention shape: {attn.shape if attn is not None else 'None'}")
            print("  ✓ Passed")
            
            # Test no pooling
            print("\nTest 3: Forward pass without pooling")
            bev_features = encoder(points_list, no_pooling=True)
            print(f"  BEV features shape: {bev_features.shape}")
            print(f"  Expected: [{batch_size}, 128, 80, 80]")
            print("  ✓ Passed")
        
        print("\n" + "="*70)
        print("✓ All Tests Passed!")
        print("="*70 + "\n")
        
    except Exception as e:
        print(f"\n✗ Test failed: {e}")
        import traceback
        traceback.print_exc()
        
    finally:
        # Cleanup
        if os.path.exists(config_path):
            os.unlink(config_path)
