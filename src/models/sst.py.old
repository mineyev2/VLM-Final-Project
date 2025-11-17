"""
SST Wrapper for LidarCLIP Integration
Compatible with mmdet3d 1.4.0, mmcv 2.1.0

Simplified approach: Use DynamicVoxelNet's extract_feat() method
which handles voxelization internally.
"""
import sys
from pathlib import Path

import torch
import torch.nn as nn
from mmengine.config import Config
from mmengine.registry import init_default_scope

# Ensure project root is in path
project_root = Path(__file__).parent.parent.parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

# Initialize mmdet3d scope
init_default_scope('mmdet3d')

# Import mmdet3d registry
from mmdet3d.registry import MODELS as MMDET3D_MODELS


def build_sst(sst_config):
    """Build SST encoder from config (uses DynamicVoxelNet)."""
    
    # Load config
    cfg = Config.fromfile(sst_config)
    
    print(f"[SST] Building model from config: {sst_config}")
    print(f"[SST] Using DynamicVoxelNet.extract_feat() method")
    
    # Add dummy bbox_head if missing (we won't use it)
    if 'bbox_head' not in cfg.model or cfg.model.bbox_head is None:
        print("[SST] Adding dummy bbox_head (required for detector, won't be used)...")
        cfg.model.bbox_head = dict(
            type='Anchor3DHead',
            num_classes=1,
            in_channels=256,
            feat_channels=256,
            use_direction_classifier=False,
            anchor_generator=dict(
                type='AlignedAnchor3DRangeGenerator',
                ranges=[[0, 0, 0, 1, 1, 1]],
                sizes=[[1, 1, 1]],
                rotations=[0],
            ),
            bbox_coder=dict(type='DeltaXYZWLHRBBoxCoder'),
            loss_cls=dict(type='FocalLoss', use_sigmoid=True, loss_weight=1.0),
            loss_bbox=dict(type='L1Loss', loss_weight=1.0),
        )
    
    # Build full detector (but we'll only use extract_feat)
    # DynamicVoxelNet.extract_feat handles voxelization internally
    model = MMDET3D_MODELS.build(cfg.model)
    
    print("[SST] ✓ Built DynamicVoxelNet (encoder part only)")
    
    return model


class LidarEncoderSST(nn.Module):
    """
    SST-based LiDAR encoder wrapper for LidarCLIP.
    Uses DynamicVoxelNet's extract_feat which handles voxelization internally.
    """
    
    def __init__(self, sst_config, clip_embedding_dim=768, checkpoint=None):
        super().__init__()
        
        print("\n" + "="*70)
        print("[LidarEncoderSST] Initializing")
        print("="*70)
        print(f"Config: {sst_config}")
        print(f"Output dim: {clip_embedding_dim}")
        print(f"Mode: Using DynamicVoxelNet encoder")
        
        # Build SST using DynamicVoxelNet
        self._sst = build_sst(sst_config)
        
        # Extract backbone output shape from config
        cfg = Config.fromfile(sst_config)
        backbone_config = cfg.model.get('backbone', {})
        backbone_output_shape = backbone_config.get('output_shape', [80, 80])
        backbone_channels = backbone_config.get('conv_out_channel', 128)
        
        print(f"\n[LidarEncoderSST] Building attention pooler...")
        print(f"  Input: [{backbone_channels}, {backbone_output_shape[0]}, {backbone_output_shape[1]}]")
        print(f"  Output: {clip_embedding_dim}")
        
        # Import attention pooler
        from src.models.attention_pool import AttentionPool2d
        
        self._pooler = AttentionPool2d(
            spacial_dim=backbone_output_shape[0],
            embed_dim=clip_embedding_dim,
            num_heads=8,
            input_dim=backbone_channels,
            output_dim=clip_embedding_dim,
        )
        
        # Load checkpoint if provided
        if checkpoint:
            print(f"\n[LidarEncoderSST] Loading checkpoint: {checkpoint}")
            self._load_checkpoint(checkpoint)
        
        print("\n" + "="*70)
        print("[LidarEncoderSST] ✓ Initialization complete!")
        print("="*70 + "\n")
    
    def _load_checkpoint(self, checkpoint_path):
        """Load pretrained weights from LidarCLIP checkpoint."""
        # Load checkpoint file
        checkpoint = torch.load(checkpoint_path, map_location='cpu')
        
        # Handle different checkpoint formats
        if 'state_dict' in checkpoint:
            state_dict = checkpoint['state_dict']
        elif 'model' in checkpoint:
            state_dict = checkpoint['model']
        else:
            state_dict = checkpoint
        
        # Strip 'lidar_encoder.' prefix and filter pooler weights
        new_state_dict = {}
        prefix_to_remove = 'lidar_encoder.'
        
        for key, value in state_dict.items():
            # Remove prefix
            if key.startswith(prefix_to_remove):
                new_key = key[len(prefix_to_remove):]
            else:
                new_key = key
            
            # Skip pooler weights (dimension mismatch with 1024 vs 768)
            if new_key.startswith('_pooler.'):
                continue
            
            new_state_dict[new_key] = value
        
        # Load state dict
        missing_keys, unexpected_keys = self.load_state_dict(
            new_state_dict, strict=False
        )
        
        # Report status
        print(f"\n[Checkpoint Loading Report]")
        print(f"  Total keys in checkpoint: {len(new_state_dict)}")
        print(f"  Loaded successfully: {len(new_state_dict) - len(unexpected_keys)}")
        print(f"  Missing (random init): {len(missing_keys)}")
        print(f"  Unexpected (ignored): {len(unexpected_keys)}")
        
        # Check critical components
        sst_loaded = sum(1 for k in new_state_dict if k.startswith('_sst.'))
        print(f"\n[Component Status]")
        print(f"  SST encoder weights: {sst_loaded} loaded ✓")
        print(f"  Pooler weights: Skipped (will be randomly initialized)")
        
        if sst_loaded > 0:
            print("\n[LidarEncoderSST] ✓ Checkpoint loaded (SST encoder has pretrained weights!)")
        else:
            print("\n[LidarEncoderSST] ⚠️  No SST weights found in checkpoint")
    
    def forward(self, point_clouds, no_pooling=False, return_attention=False):
        """
        Forward pass through SST encoder and attention pooling.
        
        Args:
            point_clouds (list[Tensor]): List of point clouds, each [N_i, 4]
            no_pooling (bool): If True, return BEV features without pooling
            return_attention (bool): If True, also return attention weights
            
        Returns:
            Tensor: Pooled features [B, clip_embedding_dim]
            Optional[Tensor]: Attention weights if return_attention=True
        """
        # Extract BEV features using DynamicVoxelNet's extract_feat
        # This handles voxelization internally
        bev_features = self._sst.extract_feat(point_clouds)
        
        # extract_feat returns a tuple, take first element
        if isinstance(bev_features, (list, tuple)):
            bev_features = bev_features[0]  # [B, C, H, W]
        
        # If no pooling requested, return spatial features
        if no_pooling:
            if return_attention:
                return bev_features, None
            else:
                return bev_features
        
        # Apply attention pooling to get CLIP-compatible embeddings
        pooled_features, attention_weights = self._pooler(
            bev_features,
            no_pooling=False,
            return_attention=return_attention
        )
        
        if return_attention:
            return pooled_features, attention_weights
        else:
            return pooled_features


# Test code
if __name__ == "__main__":
    print("="*70)
    print("Testing LidarEncoderSST")
    print("="*70)
    
    # Initialize model
    model = LidarEncoderSST(
        sst_config="src/models/mmdet3d/configs/sst_encoder_only_config.py",
        clip_embedding_dim=768,  # ViT-L/14
        checkpoint=None  # Set to checkpoint path to test loading
    )
    
    # Test with dummy data
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = model.to(device)
    
    # Create dummy point clouds (4 clouds, ~100 points each)
    point_clouds = [
        torch.randn(100, 4).to(device) 
        for _ in range(4)
    ]
    
    print("\nRunning forward pass...")
    with torch.no_grad():
        output = model(point_clouds)
    
    print(f"\n✓ Success!")
    print(f"  Input: {len(point_clouds)} point clouds")
    print(f"  Output shape: {output.shape}")
    print(f"  Expected: [4, 768]")
    
    assert output.shape == torch.Size([4, 768]), "Output shape mismatch!"
    print("\n✓ All tests passed!")
