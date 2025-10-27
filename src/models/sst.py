# =============================================================================
# sst.py – OpenMMLab v2 (MMEngine/MMCV2/mmdet3d>=1.0) compatible SST wrapper
# =============================================================================
# CORRECTED: Handles config passed as string path and extracts backbone dynamically
#
# Key features:
#   1) Imports custom models to ensure registration
#   2) Extracts backbone from full detector configs
#   3) Dynamically extracts pooler config from loaded backbone config
#   4) Proper error handling and logging
# =============================================================================

from __future__ import annotations
from typing import Any, Dict, List, Optional, Sequence, Tuple, Union

import torch
from torch import nn

# v2 stack: mmengine config + default scope
from mmengine.config import Config
from mmengine.registry import init_default_scope

# v2 stack: build models via the registry
from mmdet3d.registry import MODELS as MMDET3D_MODELS

# Optional: checkpoint loader (v2)
try:
    from mmengine.runner import load_checkpoint
except Exception:  # pragma: no cover
    load_checkpoint = None  # type: ignore

# Project-local imports
from .attention_pool import AttentionPool2d


# =============================================================================
# Utilities
# =============================================================================

def _as_config(cfg_like: Union[str, Dict[str, Any], Config]) -> Config:
    """Normalize input to an mmengine.Config.

    Args:
        cfg_like: Path to config file, dict-like, or already a Config.

    Returns:
        Config: A normalized mmengine.Config instance.
    """
    if isinstance(cfg_like, Config):
        return cfg_like
    if isinstance(cfg_like, str):
        print(f"[LiDAR Encoder] Loading config from: {cfg_like}")
        return Config.fromfile(cfg_like)
    if isinstance(cfg_like, dict):
        print(f"[LiDAR Encoder] Using dict config")
        return Config(cfg_like)
    raise TypeError(
        "Config must be a path, dict, or mmengine.config.Config"
    )

def _build_model_v2(cfg: Config):
    """Build an MMDetection3D model using the v2 registry (mmdet3d >= 1.0).

    Strategy for handling new mmdet3d versions:
    1. Initialize registry scope to resolve custom models
    2. Import mmdet3d models to trigger registration of all custom types (SSTv2, etc)
    3. Extract backbone if full detector config is provided (for encoder-only use)
    4. Build backbone directly, falling back to full detector if needed
    """
    # Initialize the registry scope
    try:
        init_default_scope('mmdet3d')
    except Exception as e:
        print(f"[LiDAR Encoder] Warning: Could not init default scope: {e}")

    # --- CRITICAL: Import mmdet3d.models to register all custom model types ---
    try:
        import mmdet3d.models  # noqa: F401
        print(f"[LiDAR Encoder] Custom models imported and registered")
    except Exception as e:
        print(f"[LiDAR Encoder] Warning: Could not import mmdet3d.models: {e}")

    m = cfg.model
    
    # --- v1 -> v2 compatibility: Convert legacy voxel_layer to data_preprocessor.voxelize_cfg ---
    if 'voxel_layer' in m:
        print("[LiDAR Encoder] Converting legacy voxel_layer to data_preprocessor format...")
        legacy_voxel = m.pop('voxel_layer')
        dp = m.setdefault('data_preprocessor', dict(type='Det3DDataPreprocessor'))
        vc = dp.setdefault('voxelize_cfg', {})
        if isinstance(legacy_voxel, dict):
            vc.update(legacy_voxel)
        print("[LiDAR Encoder] Config converted to v2 format")
    
    # --- Strategy 1: Extract backbone if config is a full detector ---
    detector_types = ['VoxelNet', 'DynamicVoxelNet', 'PointVoxelNet', 'SECOND', 
                      'PV-RCNN', 'PartA2', 'PVT']
    
    if m.get('type') in detector_types and 'backbone' in m:
        print(f"[LiDAR Encoder] Detected full detector config (type={m['type']})")
        print(f"[LiDAR Encoder] Extracting backbone for encoder-only use...")
        
        backbone_cfg = m['backbone']
        backbone_type = backbone_cfg.get('type', 'unknown')
        print(f"[LiDAR Encoder] Building backbone: {backbone_type}")
        
        try:
            model = MMDET3D_MODELS.build(backbone_cfg)
            print(f"[LiDAR Encoder] Successfully built backbone: {backbone_type}")
            if hasattr(model, 'init_weights'):
                try:
                    model.init_weights()
                except Exception:
                    pass
            model._backbone_cfg = backbone_cfg
            return model
        except Exception as e:
            print(f"[LiDAR Encoder] Warning: Could not build backbone: {e}")
            print(f"[LiDAR Encoder] Attempting full detector build...")
            
    # --- Strategy 2: Build model directly (works for backbones/encoders configured directly) ---
    try:
        print(f"[LiDAR Encoder] Building model: {m.get('type', 'unknown')}")
        model = MMDET3D_MODELS.build(m)
        print(f"[LiDAR Encoder] Successfully built model: {m.get('type', 'unknown')}")
        
        if hasattr(model, 'init_weights'):
            try:
                model.init_weights()
            except Exception:
                pass
        
        model._backbone_cfg = m
        return model
        
    except KeyError as e:
        error_msg = str(e)
        if "not in the mmdet3d::model registry" in error_msg or "not in the mmdet3d::MODELS registry" in error_msg:
            print("\n" + "="*70)
            print("[ERROR] Model Registration Failed")
            print("="*70)
            print(f"Error: {error_msg}\n")
            print("Possible causes and solutions:")
            print("  1. Custom model not installed: mmdet3d may not include SSTv2")
            print("  2. Model not imported: ensure custom models are imported")
            print("  3. Config specifies wrong model type: check your config")
            print("  4. mmdet3d version mismatch: ensure mmdet3d>=1.0")
            print("\nConfig model section:")
            print(f"  type: {m.get('type')}")
            print(f"  keys: {list(m.keys())}")
            print("="*70 + "\n")
            raise
        raise

def _maybe_load_checkpoint(model: nn.Module, checkpoint: Optional[str]) -> None:
    """Load checkpoint weights into model (with fallback for different formats)."""
    if not checkpoint:
        return
    
    print(f"[LiDAR Encoder] Loading checkpoint: {checkpoint}")
    
    if load_checkpoint is None:
        # Fallback: use torch.load for state_dict
        print("[LiDAR Encoder] Using torch.load fallback for checkpoint...")
        state = torch.load(checkpoint, map_location='cpu')
        state_dict = state.get('state_dict', state)
        model.load_state_dict(state_dict, strict=False)
        print("[LiDAR Encoder] Checkpoint loaded (non-strict)")
        return
    
    # Use mmengine's checkpoint loader
    try:
        load_checkpoint(model, checkpoint, map_location='cpu')
        print("[LiDAR Encoder] Checkpoint loaded via mmengine")
    except Exception as e:
        print(f"[LiDAR Encoder] Warning: mmengine checkpoint load failed: {e}")
        print("[LiDAR Encoder] Attempting torch.load fallback...")
        state = torch.load(checkpoint, map_location='cpu')
        state_dict = state.get('state_dict', state)
        model.load_state_dict(state_dict, strict=False)
        print("[LiDAR Encoder] Checkpoint loaded via torch.load fallback")


def _select_first_feature(feats: Any) -> torch.Tensor:
    """Normalize various feature outputs to a single Tensor.

    Accepts:
        - Tensor
        - list/tuple of Tensors (returns first)
        - dict of {name: Tensor} (returns first value by key order)
    """
    if isinstance(feats, torch.Tensor):
        return feats
    if isinstance(feats, (list, tuple)) and len(feats) > 0:
        return feats[0]
    if isinstance(feats, dict) and len(feats) > 0:
        return next(iter(feats.values()))
    raise TypeError(
        f"Unsupported feature output type: {type(feats)}"
    )


def _extract_pooler_config(backbone_cfg: Dict[str, Any]) -> Tuple[int, int]:
    """Extract pooler dimensions from backbone config.
    
    Args:
        backbone_cfg: Backbone config dict
        
    Returns:
        (spatial_dim, conv_out_channel)
    """
    # Try to get output_shape and conv_out_channel
    spatial_dim = backbone_cfg.get('output_shape')
    conv_out = backbone_cfg.get('conv_out_channel')
    
    if spatial_dim is None:
        raise ValueError(
            "backbone_cfg must have 'output_shape' key. "
            f"Available keys: {list(backbone_cfg.keys())}"
        )
    
    if conv_out is None:
        raise ValueError(
            "backbone_cfg must have 'conv_out_channel' key. "
            f"Available keys: {list(backbone_cfg.keys())}"
        )
    
    # Handle list/tuple output_shape
    if isinstance(spatial_dim, (list, tuple)):
        spatial_dim = int(spatial_dim[0])
    else:
        spatial_dim = int(spatial_dim)
    
    conv_out = int(conv_out)
    
    print(f"[LiDAR Encoder] Extracted pooler config:")
    print(f"  spatial_dim: {spatial_dim}")
    print(f"  conv_out_channel: {conv_out}")
    
    return spatial_dim, conv_out


# =============================================================================
# Public API: LidarEncoderSST
# =============================================================================

class LidarEncoderSST(nn.Module):
    """SST-based LiDAR encoder (OpenMMLab v2 compatible, mmdet3d >= 1.0).

    Parameters
    ----------
    sst_config : str | dict | mmengine.config.Config
        Path to an MMEngine-style config, a dict, or a Config. Only the
        `model` part will be used to build the encoder.
    clip_embedding_dim : int, default=512
        Projection dimension expected by the downstream CLIP text encoder.
    checkpoint : str | None, default=None
        Optional checkpoint path to load model weights.
    pool_num_heads : int, default=8
        Number of heads for the AttentionPool2d.

    Notes
    -----
    - Compatible with mmdet3d >= 1.0 (uses mmengine.Config)
    - Feature extraction tries `model.extract_feat(points, img_metas)` first,
      then falls back to `model.extract_feat(points)` if needed.
    - Backbone is extracted from full detector configs for encoder-only use.
    """

    def __init__(
        self,
        sst_config: Union[str, Dict[str, Any], Config],
        clip_embedding_dim: int = 512,
        checkpoint: Optional[str] = None,
        pool_num_heads: int = 8,
    ) -> None:
        super().__init__()

        print("\n" + "="*70)
        print("[LiDAR Encoder] Initializing LidarEncoderSST")
        print("="*70)
        
        # Load and parse config
        cfg = _as_config(sst_config)
        
        # Build SST model
        print("\n[LiDAR Encoder] Building SST model...")
        self._sst = _build_model_v2(cfg)
        
        # Load checkpoint if provided
        print("\n[LiDAR Encoder] Processing checkpoint (if provided)...")
        _maybe_load_checkpoint(self._sst, checkpoint)

        # Extract pooler dimensions from backbone config
        print("\n[LiDAR Encoder] Setting up attention pooler...")
        backbone_cfg = self._sst._backbone_cfg if hasattr(self._sst, '_backbone_cfg') else cfg.model.get('backbone', cfg.model)
        spatial_dim, conv_out = _extract_pooler_config(backbone_cfg)

        self._pooler = AttentionPool2d(
            spacial_dim=spatial_dim,
            embed_dim=clip_embedding_dim,
            num_heads=pool_num_heads,
            input_dim=conv_out,
        )
        
        print("\n" + "="*70)
        print("[LiDAR Encoder] Initialization complete!")
        print("="*70 + "\n")

    # =========================================================================

    def extract_lidar_feat(
        self,
        point_cloud: Sequence[torch.Tensor],
        img_metas: Optional[List[dict]] = None,
    ) -> torch.Tensor:
        """Extract raw grid features from the underlying SST model.

        Args:
            point_cloud: A sequence of point tensors (Ni x C, typically C in {3,4}).
            img_metas: Optional metadata list for models that expect it.

        Returns:
            A single feature Tensor (B, C, H, W).
        """
        # Try the common v2 signature first
        try:
            feats = self._sst.extract_feat(point_cloud, img_metas)  # type: ignore[arg-type]
        except TypeError:
            # Some models expose extract_feat(points) only
            feats = self._sst.extract_feat(point_cloud)  # type: ignore[misc]
        return _select_first_feature(feats)

    def forward(
        self,
        point_cloud: Sequence[torch.Tensor],
        no_pooling: bool = False,
        return_attention: bool = False,
        img_metas: Optional[List[dict]] = None,
    ) -> Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
        """Forward pass: extract and pool LiDAR features.

        Args:
            point_cloud: List of point tensors per batch item.
            no_pooling: If True, bypass attention pooling and return grid
                        features directly (B, C, H, W).
            return_attention: If True, also return the attention weights.
            img_metas: Optional mmdet-style image metadata.

        Returns:
            If return_attention is False: pooled features (B, D)
            If return_attention is True: (pooled features (B, D), attn weights)
        """
        lidar_feat = self.extract_lidar_feat(point_cloud, img_metas)  # (B, C, H, W)

        # Attention pooling
        pooled, attn = self._pooler(lidar_feat, no_pooling, return_attention)
        if return_attention:
            return pooled, attn
        return pooled


# =============================================================================
# Quick test
# =============================================================================
if __name__ == "__main__":
    import os

    cfg_path = os.environ.get("SST_CFG", "sst_encoder_only_config.py")
    
    print("="*70)
    print("LidarEncoderSST Smoke Test (mmdet3d v2 compatible)")
    print("="*70)

    try:
        encoder = LidarEncoderSST(cfg_path, clip_embedding_dim=512)
        device = "cuda" if torch.cuda.is_available() else "cpu"
        encoder.to(device).eval()

        batch_size = 2
        points = [torch.rand(10000, 4, device=device) for _ in range(batch_size)]

        print("\nRunning forward pass...")
        with torch.no_grad():
            pooled = encoder(points)
            print(f"✓ Pooled features shape: {pooled.shape}")
            
            pooled_with_attn, attn = encoder(points, return_attention=True)
            print(f"✓ Attention weights shape: {attn.shape}")
        
        print("\n" + "="*70)
        print("Success! LidarEncoderSST is working correctly.")
        print("="*70)
        
    except Exception as e:
        print(f"\n✗ Error during test: {e}")
        import traceback
        traceback.print_exc()
