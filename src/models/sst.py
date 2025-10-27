# 改动1：Config 改到 mmengine
from mmengine.config import Config

# 改动2：构建模型做了前向兼容（优先 2.x 注册器，回退到旧 build_model）
try:
    from mmdet3d.registry import MODELS  # mmdet3d 1.x/2.x
    def _build_model_from_cfg(model_cfg, train_cfg=None, test_cfg=None):
        return MODELS.build(model_cfg)
except Exception:
    from mmdet3d.models import build_model as _legacy_build_model  # 旧版
    def _build_model_from_cfg(model_cfg, train_cfg=None, test_cfg=None):
        return _legacy_build_model(model_cfg, train_cfg=train_cfg, test_cfg=test_cfg)

import torch
from torch import nn

from .attention_pool import AttentionPool2d
from .sst_encoder_only_config import model as sst_model_conf


def build_sst(config_path):
    cfg = Config.fromfile(config_path)
    model = _build_model_from_cfg(cfg.model, train_cfg=cfg.get("train_cfg"), test_cfg=cfg.get("test_cfg"))
    # 一些新版本会在 Runner/Checkpoint 时初始化；这里保留以兼容旧版
    if hasattr(model, "init_weights"):
        model.init_weights()
    return model


class LidarEncoderSST(nn.Module):
    def __init__(self, sst_config_path, clip_embedding_dim=512):
        super().__init__()
        self._sst = build_sst(sst_config_path)
        self._pooler = AttentionPool2d(
            spacial_dim=sst_model_conf["backbone"]["output_shape"][0],
            embed_dim=clip_embedding_dim,
            num_heads=8,
            input_dim=sst_model_conf["backbone"]["conv_out_channel"],
        )

    def forward(self, point_cloud, no_pooling=False, return_attention=False):
        # 不同版本的 mmdet3d 返回值可能是 tuple/list；这里沿用原写法
        lidar_features = self._sst.extract_feat(point_cloud, None)[0]  # bs, d, h, w
        pooled_feature, attn_weights = self._pooler(lidar_features, no_pooling, return_attention)
        return pooled_feature, attn_weights


if __name__ == "__main__":
    model = LidarEncoderSST("sst_encoder_only.py")
    model.to("cuda")
    points = [torch.rand(100, 4).cuda() for _ in range(16)]
    out, _ = model(points)
    print(out.shape)
