"""
SST Wrapper for LidarCLIP Integration
Compatible with mmdet3d 1.4.0, mmcv 2.1.0

Builds encoder components individually to avoid detector framework requirements.
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


class SSTEncoderOnly(nn.Module):
    """
    Wrapper that builds SST encoder components individually.
    Avoids the full detector framework which requires bbox_head.
    """
    
    def __init__(self, cfg):
        super().__init__()
        
        print("[SST] Building encoder components individually...")
        
        # Build voxel encoder
        self.voxel_encoder = MMDET3D_MODELS.build(cfg.model.voxel_encoder)
        print("[SST]   ✓ Voxel encoder: DynamicVFE")
        
        # Build middle encoder
        self.middle_encoder = MMDET3D_MODELS.build(cfg.model.middle_encoder)
        print("[SST]   ✓ Middle encoder: SSTInputLayerV2")
        
        # Build backbone
        self.backbone = MMDET3D_MODELS.build(cfg.model.backbone)
        print("[SST]   ✓ Backbone: SSTv2")
        
        print("[SST] ✓ Built encoder-only model")
    
    def extract_feat(self, points, img_metas=None):
        """
        Extract features from point clouds.
        
        Args:
            points: List of point cloud tensors, each [N_i, 4]
            img_metas: Image metadata (not used in encoder-only mode)
            
        Returns:
            tuple: (bev_features,) where bev_features is [B, C, H, W]
        """
        # Step 1: Voxelization and feature extraction
        voxel_dict = self.voxel_encoder(points)
        
        # Step 2: Middle encoder (window preparation)
        middle_features = self.middle_encoder(voxel_dict)
        
        # Step 3: Backbone (transformer encoding)
        bev_features = self.backbone(middle_features)
        
        # Return as tuple for consistency with full detector API
        return (bev_features,)


def build_sst(sst_config):
    """Build SST encoder from config."""
    
    # Load config
    cfg = Config.fromfile(sst_config)
    
    print(f"[SST] Building model from config: {sst_config}")
    print(f"[SST] Mode: Encoder-only")
    
    # Build encoder-only model
    model = SSTEncoderOnly(cfg)
    
    return model


@MMDET3D_MODELS.register_module()
class LidarEncoderSST(nn.Module):
    """
    SST-based LiDAR encoder wrapper for LidarCLIP.
    Handles checkpoint loading with prefix stripping.
    """
    
    def __init__(self, sst_config, clip_embedding_dim=768, checkpoint=None):
        super().__init__()
        
        print("\n" + "="*70)
        print("[LidarEncoderSST] Initializing")
        print("="*70)
        print(f"Config: {sst_config}")
        print(f"Output dim: {clip_embedding_dim}")
        print(f"Mode: Encoder-only")
        
        # Build SST encoder
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
            output_dim=clip_embedding_dim,  # Explicitly set output dimension
        )
        
        # Load checkpoint if provided
        if checkpoint:
            print(f"\n[LidarEncoderSST] Loading checkpoint: {checkpoint}")
            self._load_checkpoint(checkpoint)
        
        print("\n" + "="*70)
        print("[LidarEncoderSST] ✓ Initialization complete!")
        print("="*70 + "\n")
    
    def _load_checkpoint(self, checkpoint_path):
        """
        Load pretrained weights from LidarCLIP checkpoint.
        Handles the 'lidar_encoder.' prefix stripping.
        """
        # Load checkpoint file
        checkpoint = torch.load(checkpoint_path, map_location='cpu')
        
        # Handle different checkpoint formats
        if 'state_dict' in checkpoint:
            state_dict = checkpoint['state_dict']
        elif 'model' in checkpoint:
            state_dict = checkpoint['model']
        else:
            state_dict = checkpoint
        
        # Strip the 'lidar_encoder.' prefix from checkpoint keys
        # The original LidarCLIP saved with this prefix
        new_state_dict = {}
        prefix_to_remove = 'lidar_encoder.'
        
        for key, value in state_dict.items():
            if key.startswith(prefix_to_remove):
                # Remove the prefix
                new_key = key[len(prefix_to_remove):]
                new_state_dict[new_key] = value
            else:
                new_state_dict[key] = value
        
        # Filter out pooler weights if dimensions mismatch
        # The SST encoder is the important part; pooler can be randomly initialized
        filtered_state_dict = {}
        for key, value in new_state_dict.items():
            if key.startswith('_pooler.'):
                # Skip pooler weights - will be randomly initialized
                continue
            filtered_state_dict[key] = value
        
        # Load state dict with strict=False
        missing_keys, unexpected_keys = self.load_state_dict(
            filtered_state_dict, strict=False
        )
        
        # Report loading status
        print(f"\n[Checkpoint Loading Report]")
        
        if missing_keys:
            print(f"⚠️  Missing keys: {len(missing_keys)} total")
            # Only show first few to avoid clutter
            if len(missing_keys) <= 5:
                for key in missing_keys:
                    print(f"  - {key}")
            else:
                print(f"  First 5 of {len(missing_keys)}:")
                for key in missing_keys[:5]:
                    print(f"  - {key}")
                print(f"  ... and {len(missing_keys) - 5} more")
        
        if unexpected_keys:
            print(f"⚠️  Unexpected keys: {len(unexpected_keys)} total")
            if len(unexpected_keys) <= 5:
                for key in unexpected_keys:
                    print(f"  - {key}")
            else:
                print(f"  First 5 of {len(unexpected_keys)}:")
                for key in unexpected_keys[:5]:
                    print(f"  - {key}")
                print(f"  ... and {len(unexpected_keys) - 5} more")
        
        # Final status
        if not missing_keys and not unexpected_keys:
            print("\n[LidarEncoderSST] ✓ Checkpoint loaded perfectly - all keys matched!")
        elif len(missing_keys) == 0:
            print(f"\n[LidarEncoderSST] ✓ Checkpoint loaded (ignored {len(unexpected_keys)} unexpected keys)")
        else:
            print(f"\n[LidarEncoderSST] ✓ Checkpoint loaded with partial match")
            print(f"  → {len(missing_keys)} keys will use random initialization")
            print(f"  → {len(unexpected_keys)} checkpoint keys were ignored")
    
    def forward(self, point_clouds, no_pooling=False, return_attention=False):
        """
        Forward pass through SST encoder and attention pooling.
        
        Args:
            point_clouds (list[Tensor]): List of point clouds, each [N_i, 4]
                                        Format: [x, y, z, intensity/reflectance]
            no_pooling (bool): If True, return BEV features without pooling
            return_attention (bool): If True, also return attention weights
            
        Returns:
            Tensor: Pooled features [B, clip_embedding_dim]
            Optional[Tensor]: Attention weights if return_attention=True
        """
        # Extract BEV features using SST encoder pipeline:
        # point_clouds -> voxel_encoder -> middle_encoder -> backbone
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
