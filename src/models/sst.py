# ============================================================================
# sst.py — MMEngine/MMCV 2.x friendly SST wrapper (keeps original logic)
#
# - Uses mmengine Config and the mmdet3d v2 registry to build models.
# - Falls back to legacy v1 builder only if needed (for old stacks).
# - Initializes default scope ('mmdet3d') so registries resolve correctly.
# - (Nice-to-have) Patches mmcv.utils.ext_loader.load_ext to provide stub
#   functions when compiled CUDA/C++ ops are missing, preventing ImportError
#   at import time. If you actually invoke those ops, you'll still get a clear
#   NotImplementedError telling you to install the proper mmcv wheel.
# - Backward-compatible arg aliases: sst_config_path / sst_ckpt_path.
# - Does NOT change outputs; just a thin build+forward wrapper.
# ============================================================================

from __future__ import annotations
from typing import Any, Dict, Optional, Union, List

# ----------------------------------------------------------------------------
# 0) Make a real module for 'mmcv._ext' and patch ext_loader.load_ext
# ----------------------------------------------------------------------------
import sys
import types

# Ensure 'mmcv._ext' exists as a real module (importlib looks in sys.modules)
if 'mmcv._ext' not in sys.modules:
    sys.modules['mmcv._ext'] = types.ModuleType('mmcv._ext')

# Patch mmcv.utils.ext_loader.load_ext to provide stubbed functions on demand
try:
    from mmcv.utils import ext_loader as _ext_loader_mod  # type: ignore
    _orig_load_ext = _ext_loader_mod.load_ext

    def _safe_stub_func(fname: str):
        def _stub(*args, **kwargs):
            raise NotImplementedError(
                f"mmcv C++/CUDA op '{fname}' is unavailable in this environment. "
                "Install a wheel of mmcv with compiled ops matching your Torch/CUDA."
            )
        _stub.__name__ = fname
        return _stub

    def _patched_load_ext(name: str, funcs):
        """
        Try the original loader; if it fails, create/fill a stub module
        (e.g., mmcv._ext) with the requested function names so import succeeds.
        """
        try:
            return _orig_load_ext(name, funcs)
        except Exception:
            modname = f"mmcv.{name}" if not name.startswith('mmcv.') else name
            mod = sys.modules.get(modname)
            if mod is None:
                mod = types.ModuleType(modname)
                sys.modules[modname] = mod
            # Ensure every requested function exists on the module
            if isinstance(funcs, (list, tuple)):
                for f in funcs:
                    if not hasattr(mod, f):
                        setattr(mod, f, _safe_stub_func(f))
            elif isinstance(funcs, str):
                if not hasattr(mod, funcs):
                    setattr(mod, funcs, _safe_stub_func(funcs))
            return mod

    _ext_loader_mod.load_ext = _patched_load_ext  # type: ignore[attr-defined]
except Exception:
    # If mmcv isn't present yet or API differs, we just proceed; the real import
    # will likely work, or you'll see a clearer error.
    pass

# ----------------------------------------------------------------------------
# 1) MMEngine / MMCV2 config + default scope
# ----------------------------------------------------------------------------
try:
    from mmengine.config import Config  # preferred in mmcv>=2 stack
except Exception:  # pragma: no cover
    from mmcv import Config  # legacy fallback

# Initialize default scope for mmdet3d (helps registry find components)
try:
    from mmengine.registry import init_default_scope
    try:
        init_default_scope('mmdet3d')
    except Exception:
        pass
except Exception:
    pass

# ----------------------------------------------------------------------------
# 2) Build helpers: prefer the new registry; fallback to legacy builder
# ----------------------------------------------------------------------------
def _build_model_from_cfg(model_cfg: Dict[str, Any],
                          train_cfg: Optional[Dict[str, Any]] = None,
                          test_cfg: Optional[Dict[str, Any]] = None):
    # New (v2) path
    try:
        from mmdet3d.registry import MODELS  # new MMEngine registry
        return MODELS.build(model_cfg)
    except Exception:
        # Legacy (v1) fallback
        from mmdet3d.models import build_model as legacy_build_model  # type: ignore
        return legacy_build_model(model_cfg, train_cfg=train_cfg, test_cfg=test_cfg)


def _load_checkpoint_mmengine(model, checkpoint_path: str):
    try:
        from mmengine.runner import load_checkpoint  # type: ignore
        load_checkpoint(model, checkpoint_path, map_location='cpu')
        return True
    except Exception:
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
# 3) SST wrapper — original logic: build + pass-through calls
# ----------------------------------------------------------------------------
import torch
import torch.nn as nn


class LidarEncoderSST(nn.Module):
    """
    Build an SST-like LiDAR encoder from an MMDetection3D config and call it.

    Init args (both styles accepted):
      - config: str | dict | Config
      - checkpoint: Optional[str]
      - sst_config_path: Optional[str]   # alias for 'config'
      - sst_ckpt_path: Optional[str]     # alias for 'checkpoint'
      - init_weights: bool = True

    Behavior:
      - Uses mmengine registry when possible, legacy builder otherwise.
      - Optionally loads a checkpoint.
      - `extract_feat` / `forward` just delegate to the underlying model.
    """

    def __init__(self,
                 config: Union[str, Dict[str, Any], Config, None] = None,
                 checkpoint: Optional[str] = None,
                 init_weights: bool = True,
                 # Backward-compat aliases used by your caller
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
            self.cfg = Config(dict(config))
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

        # Initialize weights if available (no-op for many MMDet3D models)
        if init_weights and hasattr(self.model, 'init_weights'):
            try:
                self.model.init_weights()
            except Exception:
                pass

        # Load checkpoint if provided
        if checkpoint:
            _load_checkpoint_mmengine(self.model, checkpoint)

    # -------------------------- Public API ---------------------------------

    def extract_feat(self, *args, **kwargs):
        """Call through to the underlying model, preferring `extract_feat`."""
        if hasattr(self.model, 'extract_feat'):
            return self.model.extract_feat(*args, **kwargs)  # type: ignore
        if hasattr(self.model, 'backbone'):
            return self.model.backbone(*args, **kwargs)      # type: ignore
        return self.model(*args, **kwargs)                   # type: ignore

    def forward(self,
                points,  # e.g., List[Tensor] with (Ni, C) point clouds
                img_metas: Optional[List[dict]] = None,
                *args, **kwargs):
        """
        Preserve original behavior:
        - If you used `model.extract_feat(points, img_metas)`, keep doing that.
        - Otherwise call `model(points, img_metas, ...)` directly.
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
# 4) (Optional) smoke test
# ----------------------------------------------------------------------------
if __name__ == '__main__':
    import os
    cfg_path = os.environ.get('SST_CFG', 'your_sst_config.py')

    enc = LidarEncoderSST(sst_config_path=cfg_path, sst_ckpt_path=None)
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    enc.to(device).eval()

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
