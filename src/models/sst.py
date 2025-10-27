# ============================================================================
# sst.py  —  mmcv 2.x compatibility + minimal vector output for LidarCLIP
#
# - Stubs mmcv._ext to avoid ModuleNotFoundError with mmcv>=2.0
# - Uses mmengine.Config (fallback to mmcv.Config if needed)
# - Builds models via new mmdet3d registry (fallback to legacy builder)
# - Backward-compatible init args:
#       sst_config_path (alias for `config`)
#       sst_ckpt_path   (alias for `checkpoint`)
# - forward(points) returns (features, None) where features is [B, clip_embedding_dim]
#   using simple mean pooling and a lazy Linear projector if needed.
# ============================================================================

from __future__ import annotations
from typing import Any, Dict, Optional, Union, List, Tuple

# ----------------------------------------------------------------------------
# 1) mmcv._ext shim (prevents "ModuleNotFoundError: mmcv._ext" on mmcv>=2.0)
# ----------------------------------------------------------------------------
import mmcv  # noqa: E402
if not hasattr(mmcv, "_ext"):
    class _ExtStub:
        """Empty stub so legacy 'mmcv._ext' imports don't crash."""
        pass
    mmcv._ext = _ExtStub()

# ----------------------------------------------------------------------------
# 2) Config compatibility: prefer mmengine.Config (mmcv.Config fallback)
# ----------------------------------------------------------------------------
try:
    from mmengine.config import Config  # mmcv>=2 stack
except Exception:  # pragma: no cover
    from mmcv import Config  # legacy

# ----------------------------------------------------------------------------
# 3) Model builder compatibility: prefer 2.x registry, fallback to 1.x API
# ----------------------------------------------------------------------------
def _build_model_from_cfg(model_cfg: Dict[str, Any],
                          train_cfg: Optional[Dict[str, Any]] = None,
                          test_cfg: Optional[Dict[str, Any]] = None):
    """
    Try MMDetection3D 2.x registry first, otherwise use legacy builder.
    """
    try:
        from mmdet3d.registry import MODELS  # new registry API
        return MODELS.build(model_cfg)
    except Exception:
        # Legacy path for older MMDetection3D
        from mmdet3d.models import build_model as legacy_build_model
        return legacy_build_model(model_cfg, train_cfg=train_cfg, test_cfg=test_cfg)

# ----------------------------------------------------------------------------
# 4) SST wrapper (keeps your original logic; adds tiny pooling/projection)
# ----------------------------------------------------------------------------
import torch
import torch.nn as nn


class LidarEncoderSST(nn.Module):
    """
    Thin wrapper around your SST config/model.

    Accepted init args (both styles work):
      - config: str | dict | Config         (preferred)
      - checkpoint: Optional[str]
      - clip_embedding_dim: Optional[int]   (target vector dim; e.g., CLIP hidden size)
      - sst_config_path: Optional[str]      (alias for 'config')
      - sst_ckpt_path: Optional[str]        (alias for 'checkpoint')
      - init_weights: bool = True

    We do NOT alter model internals; we only:
      - call the underlying model/extract_feat,
      - mean-pool to a vector if needed, and
      - (lazily) project to `clip_embedding_dim` when provided.
    forward(...) returns (features, None) for compatibility with callers that
    expect `(features, attn_weights)`.
    """

    def __init__(self,
                 config: Union[str, Dict[str, Any], Config, None] = None,
                 checkpoint: Optional[str] = None,
                 clip_embedding_dim: Optional[int] = None,
                 init_weights: bool = True,
                 # Backward-compat aliases expected by your caller:
                 sst_config_path: Optional[str] = None,
                 sst_ckpt_path: Optional[str] = None,
                 **_: Any):
        super().__init__()

        # Map aliases if provided
        if sst_config_path is not None and config is None:
            config = sst_config_path
        if sst_ckpt_path is not None and checkpoint is None:
            checkpoint = sst_ckpt_path

        # Load config (string path / dict / Config)
        if isinstance(config, str):
            self.cfg: Config = Config.fromfile(config)
        elif isinstance(config, dict):
            self.cfg = Config(dict(config))  # wrap into Config for consistency
        elif isinstance(config, Config):
            self.cfg = config
        else:
            raise TypeError(
                "LidarEncoderSST: 'config' (or 'sst_config_path') must be a str path, dict, or Config."
            )

        # Build model from config
        self.model = _build_model_from_cfg(
            self.cfg.model,
            train_cfg=self.cfg.get("train_cfg"),
            test_cfg=self.cfg.get("test_cfg")
        )

        # Initialize and optionally load weights
        if init_weights and hasattr(self.model, "init_weights"):
            try:
                self.model.init_weights()
            except Exception:
                # Some models either don't implement or don't require explicit init
                pass

        if checkpoint is not None:
            self._load_checkpoint(checkpoint)

        # Target embedding dimension (e.g., CLIP ViT hidden size)
        self.target_dim: Optional[int] = int(clip_embedding_dim) if clip_embedding_dim is not None else None
        self.proj: Optional[nn.Linear] = None  # created lazily once we know src dim

    def _load_checkpoint(self, checkpoint: str):
        state = torch.load(checkpoint, map_location="cpu")
        state_dict = state.get("state_dict", state)
        missing, unexpected = self.model.load_state_dict(state_dict, strict=False)
        if missing:
            print(f"[sst] Missing keys: {missing[:12]}{' ...' if len(missing) > 12 else ''}")
        if unexpected:
            print(f"[sst] Unexpected keys: {unexpected[:12]}{' ...' if len(unexpected) > 12 else ''}")

    # -------------------------- Private helpers -----------------------------

    def _call_extract_any(self, points, img_metas=None):
        """
        Keep original logic: if your underlying model exposes `extract_feat`,
        call it; otherwise try backbone; else forward.
        """
        if hasattr(self.model, "extract_feat"):
            try:
                return self.model.extract_feat(points, img_metas)  # type: ignore
            except TypeError:
                return self.model.extract_feat(points)             # type: ignore
        if hasattr(self.model, "backbone"):
            return self.model.backbone(points)                     # type: ignore
        return self.model(points)                                  # type: ignore

    def _to_batch_tensor(self, feats: torch.Tensor) -> torch.Tensor:
        """
        Normalize various shapes to (B, C) by mean pooling over non-(B,C) dims.
        Supported:
          - (B, C, H, W) -> mean over H, W
          - (B, T, C)    -> mean over T
          - (T, C)       -> add batch dim -> (1, C)
          - (C,)         -> (1, C)
        """
        if feats.ndim == 4:
            # (B, C, H, W)
            return feats.mean(dim=(2, 3))
        if feats.ndim == 3:
            # could be (B, T, C) or (C, H, W) — assume (B, T, C) is most common
            if feats.shape[0] >= 1 and feats.shape[-1] <= 4096:
                # (B, T, C) -> mean over T
                return feats.mean(dim=1)
            # fallback: treat as (B, C, T) and mean over last dim
            return feats.mean(dim=2)
        if feats.ndim == 2:
            # (T, C) -> add batch dim
            return feats.mean(dim=0, keepdim=True)
        if feats.ndim == 1:
            # (C,) -> (1, C)
            return feats.unsqueeze(0)
        raise TypeError(f"Unexpected feature tensor shape: {feats.shape}")

    def _maybe_project(self, x: torch.Tensor) -> torch.Tensor:
        """
        If target_dim is set and differs from current dim, lazily create a projector.
        """
        if self.target_dim is None:
            return x
        in_dim = x.shape[-1]
        if in_dim == self.target_dim:
            return x
        if self.proj is None:
            self.proj = nn.Linear(in_dim, self.target_dim, bias=True)
        return self.proj(x)

    # -------------------------- Public API ----------------------------------

    def extract_feat(self, *args, **kwargs):
        """
        Expose extract_feat for callers that want raw model outputs.
        """
        return self._call_extract_any(*args, **kwargs)

    def forward(self,
                points: Union[List[torch.Tensor], Any],
                img_metas: Optional[List[dict]] = None,
                *args, **kwargs) -> Tuple[torch.Tensor, None]:
        """
        Preserve original call pattern but return a vector per sample:

        Returns:
          (features, None)
            features: [B, clip_embedding_dim] if `clip_embedding_dim` was set,
                      otherwise [B, C] from pooled model output.
            None: attention weights placeholder for compatibility.
        """
        feats = self._call_extract_any(points, img_metas)

        # If the model returns a list/tuple/dict, pick a sensible tensor
        if isinstance(feats, (list, tuple)):
            feats = feats[0]
        elif isinstance(feats, dict):
            for key in ("feat", "feats", "x", "out", "neck_out"):
                if key in feats and isinstance(feats[key], torch.Tensor):
                    feats = feats[key]
                    break
            if isinstance(feats, dict):
                # still a dict — can't use
                raise TypeError(f"Unexpected feature dict keys: {list(feats.keys())}")

        if not isinstance(feats, torch.Tensor):
            raise TypeError(f"Unexpected feature type: {type(feats)}")

        # Mean-pool to (B, C)
        vec = self._to_batch_tensor(feats)
        # Optionally project to requested dim (e.g., CLIP hidden size)
        vec = self._maybe_project(vec)

        # Return attention placeholder as None (your caller expects a 2-tuple)
        return vec, None


# ----------------------------------------------------------------------------
# 5) (Optional) quick smoke test
# ----------------------------------------------------------------------------
if __name__ == "__main__":
    import os

    cfg_path = os.environ.get("SST_CFG", "your_sst_config.py")
    enc = LidarEncoderSST(sst_config_path=cfg_path, clip_embedding_dim=768)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    enc.to(device).eval()
    dummy_points = [torch.zeros((50, 4), device=device)]
    try:
        feats, attn = enc(dummy_points, None)
        print("[sst] vector shape:", tuple(feats.shape), "| attn:", attn)
    except Exception as e:
        print("[sst] Smoke test warning:", repr(e))
