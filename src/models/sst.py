# ============================================================================
# sst.py — MMEngine/MMCV 2.0 friendly SST wrapper (no extra pooling/ops)
#
# - Creates a real dummy module "mmcv._ext" in sys.modules to avoid
#   `ModuleNotFoundError: mmcv._ext` raised by legacy ext_loader paths.
# - Uses mmengine Config and the new mmdet3d registry build path.
# - Initializes default scope to 'mmdet3d' when available.
# - Backward-compatible init arg names (sst_config_path / sst_ckpt_path).
# - Keeps your original call behavior: no attention/pooling additions.
# ============================================================================

from __future__ import annotations
from typing import Any, Dict, Optional, Union, List

# ----------------------------------------------------------------------------
# 0) Provide a real dummy module for 'mmcv._ext' so importlib can find it
# ----------------------------------------------------------------------------
import sys
import types

if 'mmcv._ext' not in sys.modules:
    sys.modules['mmcv._ext'] = types.ModuleType('mmcv._ext')

# ----------------------------------------------------------------------------
# 1) MMEngine / MMCV2 imports (Config + default scope)
# ----------------------------------------------------------------------------
try:
    from mmengine.config import Config
except Exception as e:  # fallback for very old stacks
    from mmcv import Config  # type: ignore

# Initialize default scope for mmdet3d if available (MMEngine-style)
try:
    from mmengine.registry import init_default_scope
    try:
        init_default_scope('mmdet3d')
    except Exception:
        # It's okay if this fails in some envs; build may still work.
        pass
except Exception:
    pass

# ----------------------------------------------------------------------------
# 2) Model build helpers (prefer new registry; fallback to legacy)
# ----------------------------------------------------------------------------
def _build_model_from_cfg(model_cfg: Dict[str, Any],
                          train_cfg: Optional[Dict[str, Any]] = None,
                          test_cfg: Optional[Dict[str, Any]] = None):
    """
    Prefer the MMEngine registry API; fallback to legacy builder if needed.
    """
    try:
        # New (MMEngine) registry path
        from mmdet3d.registry import MODELS  # type: ignore
        return MODELS.build(model_cfg)
    except Exception:
        # Legacy builder path
        from mmdet3d.models import build_model as legacy_build_model  # type: ignore
        return legacy_build_model(model_cfg, train_cfg=train_cfg, test_cfg=test_cfg)

def _load_checkpoint_mmengine(model, checkpoint_path: str):
    """
    Load a checkpoint the MMEngine way. Handles both raw state dicts and files.
    """
    try:
        from mmengine.runner import load_checkpoint  # type: ignore
        load_checkpoint(model, checkpoint_path, map_location='cpu')
        return True
    except Exception:
        # Fallback: torch.load + load_state_dict
        import torch
        state = torch.load(checkpoint_path, map_location='cpu')
        state_dict = state.get('state_dict', state)
        missing, unexpected = model.load_state_dict(state_dict, strict=False)
        if missing:
            print(f"[sst] Missing keys: {missing[:12]}{' ...' if len(missing) > 12 else ''}")
        if unexpected:
            print(f"[sst] Unexpected keys: {unexpected[:12]}{' ...' if len(unexpected) > 12 else ''}")
        return True

# ----------------------------------------------------------------------------
# 3) SST wrapper — keeps your original usage/logic (no extra heads/pooling)
# ----------------------------------------------------------------------------
import torch
import torch.nn as nn


class LidarEncoderSST(nn.Module):
    """
    Thin wrapper for building and calling an SST-like LiDAR encoder from config.

    Accepted init args (both styles work):
      - config: str | dict | Config         (preferred)
      - checkpoint: Optional[str]
      - sst_config_path: Optional[str]      (alias for 'config')
      - sst_ckpt_path: Optional[str]        (alias for 'checkpoint')
      - init_weights: bool = True

    Behavior:
      - Builds the model from the given config using the MMEngine registry if possible.
      - Optionally loads a checkpoint.
      - `extract_feat()` and `forward()` just call through to the underlying model
        (no attention/pooling added). Whatever your model returns is returned.
    """

    def __init__(self,
                 config: Union[str, Dict[str, Any], Config, None] = None,
                 checkpoint: Optional[str] = None,
                 init_weights: bool = True,
                 # Backward-compat aliases expected by some callers:
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
            self.cfg = Config(dict(config))  # wrap in Config for consistency
        elif isinstance(config, Config):
            self.cfg = config
        else:
            raise TypeError(
                "LidarEncoderSST: 'config' (or 'sst_config_path') must be a str path, dict, or Config."
            )

        # Build model
        self.model = _build_model_from_cfg(
            self.cfg.model,
            train_cfg=self.cfg.get('train_cfg'),
            test_cfg=self.cfg.get('test_cfg')
        )

        # Initialize weights if the model supports it
        if init_weights and hasattr(self.model, 'init_weights'):
            try:
                self.model.init_weights()
            except Exception:
                # Some models don't implement or don't require explicit init
                pass

        # Load checkpoint if provided
        if checkpoint:
            _load_checkpoint_mmengine(self.model, checkpoint)

    # -------------------------- Public API ---------------------------------

    def extract_feat(self, *args, **kwargs):
        """
        If your model exposes `extract_feat`, call it. Otherwise try backbone/forward.
        """
        if hasattr(self.model, 'extract_feat'):
            return self.model.extract_feat(*args, **kwargs)  # type: ignore
        if hasattr(self.model, 'backbone'):
            return self.model.backbone(*args, **kwargs)      # type: ignore
        return self.model(*args, **kwargs)                   # type: ignore

    def forward(self,
                points,  # Often: List[Tensor] with (Ni, C) point clouds
                img_metas: Optional[List[dict]] = None,
                *args, **kwargs):
        """
        Preserve original behavior: if your code used
            model.extract_feat(points, img_metas)
        keep using it; otherwise call the model directly.
        """
        if hasattr(self.model, 'extract_feat'):
            try:
                return self.model.extract_feat(points, img_metas, *args, **kwargs)  # type: ignore
            except TypeError:
                return self.model.extract_feat(points, *args, **kwargs)             # type: ignore
        if hasattr(self.model, 'forward'):
            return self.model(points, img_metas, *args, **kwargs)  # type: ignore
        raise RuntimeError("Underlying model has no usable forward/extract path.")


# ----------------------------------------------------------------------------
# 4) (Optional) quick smoke test — run: python sst.py
# ----------------------------------------------------------------------------
if __name__ == '__main__':
    import os
    cfg_path = os.environ.get('SST_CFG', 'your_sst_config.py')

    enc = LidarEncoderSST(sst_config_path=cfg_path, sst_ckpt_path=None)
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    enc.to(device).eval()

    # Minimal dummy points: one sample with (N=10, C=4)
    dummy_points = [torch.zeros((10, 4), device=device)]
    try:
        out = enc.extract_feat(dummy_points, None)
        if isinstance(out, (list, tuple)):
            print('[sst] extract_feat returned list/tuple:',
                  [getattr(x, 'shape', type(x)) for x in out])
        elif isinstance(out, dict):
            print('[sst] extract_feat returned dict keys:', list(out.keys()))
        else:
            print('[sst] extract_feat output:', getattr(out, 'shape', type(out)))
    except Exception as e:
        print('[sst] Smoke test warning:', repr(e))
