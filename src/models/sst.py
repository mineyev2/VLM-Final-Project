# =============================================================================
# sst.py — OpenMMLab v2 (MMEngine/MMCV2/mmdet3d>=1.0) compatible SST wrapper
# =============================================================================
# Key changes vs. legacy code:
#   1) Uses mmengine.Config (NOT mmcv.Config) and the v2 registry build path:
#        from mmdet3d.registry import MODELS
#        model = MODELS.build(cfg.model)
#   2) Initializes default scope ('mmdet3d') so registries resolve correctly.
#   3) Robust feature extraction: handles list/tuple/dict/Tensor returns.
#   4) Optional checkpoint loading via mmengine.runner.load_checkpoint.
#   5) Keeps your AttentionPool2d integration unchanged.
#
# This file intentionally avoids v1-era APIs such as:
#   - from mmcv import Config
#   - from mmdet3d.models import build_model
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
from .sst_encoder_only_config import model as sst_model_conf


# -----------------------------------------------------------------------------
# Utilities
# -----------------------------------------------------------------------------

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
        return Config.fromfile(cfg_like)
    if isinstance(cfg_like, dict):
        return Config(cfg_like)
    raise TypeError(
        "Config must be a path, dict, or mmengine.config.Config"
    )

def _build_model_v2(cfg: Config):
    """Build an MMDetection3D model using the v2 registry.

    - Auto-converts legacy v1 keys (e.g., `voxel_layer`) into the v2 location
      under `data_preprocessor.voxelize_cfg`.
    """
    try:
        init_default_scope('mmdet3d')
    except Exception:
        pass

    # --- Begin v1 -> v2 compatibility shim ---
    m = cfg.model
    # Old configs sometimes have voxelization at the top-level
    for legacy_key in ("voxel_layer", "pts_voxel_layer"):
        if legacy_key in m:
            legacy_voxel = m.pop(legacy_key)
            dp = m.setdefault("data_preprocessor", dict(type="Det3DDataPreprocessor"))
            vc = dp.setdefault("voxelize_cfg", {})
            # Prefer explicit values from the legacy block
            if isinstance(legacy_voxel, dict):
                vc.update(legacy_voxel)
    # --- End shim ---

    model = MMDET3D_MODELS.build(m)

    if hasattr(model, 'init_weights'):
        try:
            model.init_weights()
        except Exception:
            pass

    return model

def _maybe_load_checkpoint(model: nn.Module, checkpoint: Optional[str]) -> None:
    if not checkpoint:
        return
    if load_checkpoint is None:
        # Fallback: torch.load state_dict (best-effort)
        state = torch.load(checkpoint, map_location='cpu')
        state_dict = state.get('state_dict', state)
        model.load_state_dict(state_dict, strict=False)
        return
    load_checkpoint(model, checkpoint, map_location='cpu')


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
        # Fetch first value deterministically
        return next(iter(feats.values()))
    raise TypeError(
        f"Unsupported feature output type: {type(feats)}"
    )


# -----------------------------------------------------------------------------
# Public API: LidarEncoderSST
# -----------------------------------------------------------------------------

class LidarEncoderSST(nn.Module):
    """SST-based LiDAR encoder (OpenMMLab v2 compatible).

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
    - This wrapper avoids all v1 (mmcv 1.x) APIs.
    - Feature extraction tries `model.extract_feat(points, img_metas)` first,
      then falls back to `model.extract_feat(points)` if needed.
    """

    def __init__(
        self,
        sst_config: Union[str, Dict[str, Any], Config],
        clip_embedding_dim: int = 512,
        checkpoint: Optional[str] = None,
        pool_num_heads: int = 8,
    ) -> None:
        super().__init__()

        cfg = _as_config(sst_config)
        self._sst = _build_model_v2(cfg)
        _maybe_load_checkpoint(self._sst, checkpoint)

        # Derive pooling dims from your config helper (kept as-is)
        # Expecting square spatial grids: output_shape[0] == output_shape[1]
        spacial_dim = int(sst_model_conf["backbone"]["output_shape"][0])
        conv_out = int(sst_model_conf["backbone"]["conv_out_channel"])

        self._pooler = AttentionPool2d(
            spacial_dim=spacial_dim,
            embed_dim=clip_embedding_dim,
            num_heads=pool_num_heads,
            input_dim=conv_out,
        )

    # ------------------------------ Forward ---------------------------------

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
        """Forward pass.

        Args:
            point_cloud: List[List[float]]-like point tensors per batch item.
            no_pooling: If True, bypass attention pooling and return the grid
                        features directly.
            return_attention: If True, also return the attention weights.
            img_metas: Optional mmdet-style image meta.

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


# -----------------------------------------------------------------------------
# Smoke test
# -----------------------------------------------------------------------------
if __name__ == "__main__":
    import os

    # Example usage: set SST_CFG to your MMEngine config file
    cfg_path = os.environ.get("SST_CFG", "sst_encoder_only.py")

    encoder = LidarEncoderSST(cfg_path, clip_embedding_dim=512)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    encoder.to(device).eval()

    # Create a fake batch of LiDAR points
    batch_size = 2
    points = [torch.rand(10000, 4, device=device) for _ in range(batch_size)]

    with torch.no_grad():
        pooled = encoder(points)  # (B, D)
        print("pooled shape:", getattr(pooled, "shape", type(pooled)))
