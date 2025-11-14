"""
SST Encoder Wrapper for LidarCLIP
Uses modernized mmdet3d components compatible with PyTorch 2.x and mmcv 2.x
"""

import os
import torch
from torch import nn
from mmengine import Config
from mmdet3d.registry import MODELS

# ✅ CRITICAL: Import SST components to register them
from .sst_v2 import SSTv2
from .sst_input_layer_v2 import SSTInputLayerV2
from .voxel_encoder import DynamicVFE

# Import attention pooling (you'll need to have this file)
try:
    from .attention_pool import AttentionPool2d
except ImportError:
    print("Warning: attention_pool.py not found. Using placeholder.")
    class AttentionPool2d(nn.Module):
        def __init__(self, spacial_dim, embed_dim, num_heads, input_dim):
            super().__init__()
            self.pool = nn.AdaptiveAvgPool2d(1)
            self.proj = nn.Linear(input_dim, embed_dim)
        
        def forward(self, x, no_pooling=False, return_attention=False):
            if no_pooling:
                return x, None
            B, C, H, W = x.shape
            pooled = self.pool(x).squeeze(-1).squeeze(-1)  # [B, C]
            out = self.proj(pooled)  # [B, embed_dim]
            return (out, None) if return_attention else out


def build_sst(config_path):
    """
    Build SST model from config file using modern mmdet3d registry.
    
    Args:
        config_path: Path to SST configuration file (.py)
    
    Returns:
        SST model instance
    """
    if not os.path.isfile(config_path):
        raise FileNotFoundError(f"Config file not found: {config_path}")
    
    # Load config using mmengine
    cfg = Config.fromfile(config_path)
    
    # Build model using MODELS registry
    if hasattr(cfg, 'model'):
        model = MODELS.build(cfg.model)
    else:
        # If config doesn't have 'model' key, assume the config IS the model config
        model = MODELS.build(cfg)
    
    # Initialize weights if method exists
    if hasattr(model, 'init_weights'):
        try:
            model.init_weights()
        except Exception as e:
            print(f"Warning: Could not initialize weights: {e}")
    
    return model


class LidarEncoderSST(nn.Module):
    """
    Wrapper for SST (Sparse Spatial Transformer) encoder with attention pooling.
    
    This class:
    1. Builds SST backbone from config
    2. Adds attention pooling to get fixed-size features
    3. Handles checkpoint loading
    
    Args:
        sst_config: Path to SST config file OR Config object
        clip_embedding_dim: Output embedding dimension (should match CLIP)
        checkpoint: Optional path to pretrained weights
    """
    
    def __init__(self, sst_config, clip_embedding_dim=1024, checkpoint=None):
        super().__init__()
        
        print("Initializing LidarEncoderSST...")
        
        # Build SST backbone
        if isinstance(sst_config, str):
            print(f"  Loading SST config from: {sst_config}")
            self._sst = build_sst(sst_config)
        else:
            # If config is already a Config object or dict
            print("  Building SST from Config object")
            self._sst = MODELS.build(sst_config)
        
        print("  ✓ SST backbone built")
        
        # Get output configuration
        # These should match your sst_encoder_only_config.py
        # Adjust if your config has different values
        if hasattr(self._sst, 'output_shape'):
            output_shape = self._sst.output_shape
        else:
            # Default values - VERIFY these match your config!
            output_shape = (180, 180)
            print(f"  Warning: Using default output_shape {output_shape}")
        
        # Get the output channels from the SST config
        if hasattr(self._sst, 'conv_out_channel'):
            sst_output_channels = self._sst.conv_out_channel
        elif hasattr(self._sst, 'd_model'):
            sst_output_channels = self._sst.d_model[-1] if isinstance(self._sst.d_model, list) else self._sst.d_model
        else:
            # Default value - VERIFY this matches your config!
            sst_output_channels = 256
            print(f"  Warning: Using default sst_output_channels {sst_output_channels}")
        
        print(f"  SST output: {sst_output_channels} channels, {output_shape} spatial")
        
        # Attention pooling to get fixed-size features aligned with CLIP
        self._pooler = AttentionPool2d(
            spacial_dim=output_shape[0],
            embed_dim=clip_embedding_dim,
            num_heads=8,
            input_dim=sst_output_channels,
        )
        print(f"  ✓ Attention pooler: {sst_output_channels} -> {clip_embedding_dim}")
        
        # Load checkpoint if provided
        if checkpoint is not None and os.path.isfile(checkpoint):
            print(f"  Loading checkpoint from: {checkpoint}")
            self.load_checkpoint(checkpoint)
        elif checkpoint is not None:
            print(f"  Warning: Checkpoint not found: {checkpoint}")
        
        print("✓ LidarEncoderSST initialized")
    
    def load_checkpoint(self, checkpoint_path):
        """
        Load pretrained weights from checkpoint.
        
        Handles different checkpoint formats:
        - PyTorch Lightning checkpoints with 'state_dict' key
        - Standard PyTorch checkpoints with 'model' key
        - Raw state dictionaries
        """
        try:
            ckpt = torch.load(checkpoint_path, map_location='cpu')
            
            # Handle different checkpoint formats
            if 'state_dict' in ckpt:
                state_dict = ckpt['state_dict']
            elif 'model' in ckpt:
                state_dict = ckpt['model']
            else:
                state_dict = ckpt
            
            # Try to load with prefix handling
            missing_keys, unexpected_keys = self.load_state_dict(state_dict, strict=False)
            
            if len(missing_keys) > 0:
                print(f"    Missing keys: {len(missing_keys)}")
            if len(unexpected_keys) > 0:
                print(f"    Unexpected keys: {len(unexpected_keys)}")
            
            print(f"    ✓ Checkpoint loaded successfully")
            
        except Exception as e:
            print(f"    ✗ Failed to load checkpoint: {e}")
            print(f"    Continuing without pretrained weights")
    
    def forward(self, point_clouds, no_pooling=False, return_attention=False):
        """
        Forward pass through SST encoder and attention pooling.
        
        Args:
            point_clouds: List of point cloud tensors, each [N, 4] (x, y, z, intensity)
                         OR dict with voxelized features (depends on your preprocessing)
            no_pooling: If True, return spatial features without pooling
            return_attention: If True, also return attention weights
        
        Returns:
            pooled_features: [B, clip_embedding_dim] if pooling
                            [B, C, H, W] if no_pooling
            attention_weights: (optional) attention weights from pooler
        """
        # Extract features using SST
        # The extract_feat method should return list of features at different scales
        # We take the first (finest) scale
        try:
            if hasattr(self._sst, 'extract_feat'):
                # Standard mmdet3d interface
                lidar_features = self._sst.extract_feat(point_clouds, None)[0]
            else:
                # Direct forward pass
                lidar_features = self._sst(point_clouds)
                if isinstance(lidar_features, (list, tuple)):
                    lidar_features = lidar_features[0]
            
            # lidar_features shape: [B, C, H, W]
            
        except Exception as e:
            print(f"Error in SST forward pass: {e}")
            print(f"Input type: {type(point_clouds)}")
            if isinstance(point_clouds, list):
                print(f"List length: {len(point_clouds)}")
                if len(point_clouds) > 0:
                    print(f"First element shape: {point_clouds[0].shape if hasattr(point_clouds[0], 'shape') else 'N/A'}")
            raise
        
        # Pool to fixed size
        pooled_features, attn_weights = self._pooler(
            lidar_features, 
            no_pooling=no_pooling,
            return_attention=return_attention
        )
        
        if return_attention:
            return pooled_features, attn_weights
        return pooled_features


# Test code
if __name__ == "__main__":
    print("Testing LidarEncoderSST...")
    
    # Example usage
    try:
        encoder = LidarEncoderSST(
            sst_config="./mmdet3d/configs/sst_encoder_only_config.py",
            clip_embedding_dim=1024,
            checkpoint=None
        )
        print("\n✓ Encoder created successfully")
        
        # Create dummy input
        B = 2
        dummy_points = [torch.randn(1000, 4) for _ in range(B)]
        
        print("\nTesting forward pass...")
        output = encoder(dummy_points)
        print(f"✓ Output shape: {output.shape}")
        print(f"  Expected: [{B}, 1024]")
        
        if output.shape == (B, 1024):
            print("\n✅ All tests passed!")
        else:
            print("\n⚠️ Output shape mismatch!")
            
    except Exception as e:
        print(f"\n✗ Test failed: {e}")
        import traceback
        traceback.print_exc()
