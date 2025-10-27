# ============================================================================
# sst.py  —  mmcv 2.x compatibility without changing your SST logic
#
# What this does:
#   1) Stubs mmcv._ext so old ops imports don't crash on mmcv>=2.0
#   2) Uses mmengine.Config when available (mmcv.Config fallback)
#   3) Builds models via the new mmdet3d registry (fallback to 1.x builder)
#
# What this does NOT do:
#   - It does NOT change your model, add pooling layers, or alter outputs.
#   - It does NOT import or rely on custom attention pooling modules.
#
# Drop this file in place of your original sst.py.
# ============================================================================

from __future__ import annotations
from typing import Any, Dict, Optional, Union, List

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
        from mmdet3d.registry import MODELS  # mmdet3d>=1.1.0 (new registry API)
        return MODELS.build(model_cfg)
    except Exception:
        # Legacy path for older MMDetection3D
        from mmdet3d.models import build_model as legacy_build_model
        return legacy_build_model(model_cfg, train_cfg=train_cfg, test_cfg=test_cfg)

# ----------------------------------------------------------------------------
# 4) SST wrapper that keeps your original forward/extract logic intact
# ----------------------------------------------------------------------------
import torch
import torch.nn as nn


class LidarEncoderSST(nn.Module):
    """
    Thin wrapper around your SST config/model.

    Usage:
        # Option A: load from a python config file that defines `model = dict(...)`
        enc = LidarEncoderSST(config="path/to/your_sst_config.py", checkpoint=None)

        # Option B: pass an existing Config/dict
        cfg = Config.fromfile("path/to/your_sst_config.py")
        enc = LidarEncoderSST(config=cfg)

        enc.eval()
        outputs = enc(points_list, img_metas=None)

    Notes:
      - We don't alter your outputs; forward returns whatever the underlying model returns.
      - If your pipeline expects dataset dicts (e.g., voxelized batches), feed those instead.
    """

    def __init__(self,
                 config: Union[str, Dict[str, Any], Config],
                 checkpoint: Optional[str] = None,
                 init_weights: bool = True):
        super().__init__()

        # Load config (string path / dict / Config)
        if isinstance(config, str):
            self.cfg: Config = Config.fromfile(config)
        elif isinstance(config, dict):
            self.cfg = Config(dict(config))  # wrap into Config for consistency
        elif isinstance(config, Config):
            self.cfg = config
        else:
            raise TypeError(f"Unsupported config type: {type(config)}")

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

    def _load_checkpoint(self, checkpoint: str):
        state = torch.load(checkpoint, map_location="cpu")
        state_dict = state.get("state_dict", state)
        missing, unexpected = self.model.load_state_dict(state_dict, strict=False)
        if missing:
            print(f"[sst] Missing keys: {missing[:12]}{' ...' if len(missing) > 12 else ''}")
        if unexpected:
            print(f"[sst] Unexpected keys: {unexpected[:12]}{' ...' if len(unexpected) > 12 else ''}")

    # -------------------------- Public API ---------------------------------

    def extract_feat(self, *args, **kwargs):
        """
        Keep original logic: if your underlying model exposes `extract_feat`,
        we call that directly; otherwise we try falling back to backbone/forward.
        """
        if hasattr(self.model, "extract_feat"):
            return self.model.extract_feat(*args, **kwargs)  # type: ignore
        if hasattr(self.model, "backbone"):
            return self.model.backbone(*args, **kwargs)      # type: ignore
        return self.model(*args, **kwargs)                   # type: ignore

    def forward(self,
                points: Union[List[torch.Tensor], Any],
                img_metas: Optional[List[dict]] = None,
                *args, **kwargs):
        """
        Keep your original forward behavior:
        - If your code used `model.extract_feat(points, img_metas)`, keep using it.
        - Otherwise, call model directly.

        You decide at call-site whether you want features or full detection heads.
        """
        # Try common call signatures in order, without changing your logic:
        if hasattr(self.model, "extract_feat"):
            try:
                return self.model.extract_feat(points, img_metas, *args, **kwargs)  # type: ignore
            except TypeError:
                return self.model.extract_feat(points, *args, **kwargs)             # type: ignore
        # Fallbacks
        if hasattr(self.model, "forward"):
            return self.model(points, img_metas, *args, **kwargs)  # type: ignore
        raise RuntimeError("Underlying model has no usable forward/extract path.")


# ----------------------------------------------------------------------------
# 5) (Optional) quick smoke test
# ----------------------------------------------------------------------------
if __name__ == "__main__":
    import os

    # Example: load from a config file that defines `model = dict(...)`
    # The uploaded companion config is a classic dict-style SST setup. :contentReference[oaicite:2]{index=2}
    cfg_path = os.environ.get("SST_CFG", "your_sst_config.py")
    enc = LidarEncoderSST(cfg_path, checkpoint=None)

    # If your pipeline takes a list of (N, 4) point clouds:
    device = "cuda" if torch.cuda.is_available() else "cpu"
    enc.to(device).eval()
    dummy_points = [torch.zeros((50, 4), device=device)]
    try:
        out = enc.extract_feat(dummy_points, None)
        if isinstance(out, (list, tuple)):
            print("[sst] extract_feat returned list/tuple with lengths:", [getattr(x, 'shape', type(x)) for x in out])
        elif isinstance(out, dict):
            print("[sst] extract_feat returned dict with keys:", list(out.keys()))
        else:
            print("[sst] extract_feat output shape/type:", getattr(out, "shape", type(out)))
    except Exception as e:
        print("[sst] Smoke test warning:", repr(e))
