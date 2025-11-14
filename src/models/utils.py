"""
Utility classes for Voxel Feature Encoding
Modernized for mmcv 2.x / mmengine compatibility
"""

import torch
from mmcv.cnn import build_norm_layer
from mmengine.runner import autocast  # ✅ UPDATED: Modern import
from torch import nn
from torch.nn import functional as F


def get_paddings_indicator(actual_num, max_num, axis=0):
    """
    Create boolean mask by actually number of a padded tensor.

    Args:
        actual_num (torch.Tensor): Actual number of points in each voxel.
        max_num (int): Max number of points in each voxel

    Returns:
        torch.Tensor: Mask indicates which points are valid inside a voxel.
    """
    actual_num = torch.unsqueeze(actual_num, axis + 1)
    # tiled_actual_num: [N, M, 1]
    max_num_shape = [1] * len(actual_num.shape)
    max_num_shape[axis + 1] = -1
    max_num = torch.arange(
        max_num, dtype=torch.int, device=actual_num.device).view(max_num_shape)
    # tiled_actual_num: [[3,3,3,3,3], [4,4,4,4,4], [2,2,2,2,2]]
    # tiled_max_num: [[0,1,2,3,4], [0,1,2,3,4], [0,1,2,3,4]]
    paddings_indicator = actual_num.int() > max_num
    # paddings_indicator shape: [batch_size, max_num]
    return paddings_indicator


class VFELayer(nn.Module):
    """
    Voxel Feature Encoder layer.

    The voxel encoder is composed of a series of these layers.
    This module does not support average pooling and only uses
    max pooling to gather features inside a VFE.

    Args:
        in_channels (int): Number of input channels.
        out_channels (int): Number of output channels.
        norm_cfg (dict): Config dict of normalization layers
        max_out (bool): Whether aggregate the features of points inside
            each voxel and only return voxel features.
        cat_max (bool): Whether concatenate the aggregated features
            and pointwise features.
    """

    def __init__(self,
                 in_channels,
                 out_channels,
                 norm_cfg=dict(type='BN1d', eps=1e-3, momentum=0.01),
                 max_out=True,
                 cat_max=True):
        super(VFELayer, self).__init__()
        self.fp16_enabled = False
        self.cat_max = cat_max
        self.max_out = max_out

        self.norm = build_norm_layer(norm_cfg, out_channels)[1]
        self.linear = nn.Linear(in_channels, out_channels, bias=False)

    @autocast(dtype=torch.float16)  # ✅ UPDATED: Modern autocast
    def forward(self, inputs):
        """
        Forward function.

        Args:
            inputs (torch.Tensor): Voxels features of shape (N, M, C).
                N is the number of voxels, M is the number of points in
                voxels, C is the number of channels of point features.

        Returns:
            torch.Tensor: Voxel features. There are three modes:
                - `max_out=False`: Return point-wise features in
                    shape (N, M, C).
                - `max_out=True` and `cat_max=False`: Return aggregated
                    voxel features in shape (N, C)
                - `max_out=True` and `cat_max=True`: Return concatenated
                    point-wise features in shape (N, M, C).
        """
        # [K, T, in_channels] -> [K, T, out_channels]
        voxel_count = inputs.shape[1]

        x = self.linear(inputs)
        x = self.norm(x.permute(0, 2, 1).contiguous()).permute(0, 2, 1).contiguous()
        pointwise = F.relu(x)
        # [K, T, out_channels]
        
        if self.max_out:
            aggregated = torch.max(pointwise, dim=1, keepdim=True)[0]
        else:
            # This is for fusion layer
            return pointwise

        if not self.cat_max:
            return aggregated.squeeze(1)
        else:
            # [K, 1, out_channels]
            repeated = aggregated.repeat(1, voxel_count, 1)
            concatenated = torch.cat([pointwise, repeated], dim=2)
            # [K, T, 2 * out_channels]
            return concatenated


class DynamicVFELayer(nn.Module):
    """
    Dynamic Voxel Feature Encoder layer.
    
    This layer is similar to VFELayer but designed for dynamic voxelization
    where voxel sizes can vary.

    Args:
        in_channels (int): Number of input channels.
        out_channels (int): Number of output channels.
        norm_cfg (dict): Config dict of normalization layers.
    """

    def __init__(self,
                 in_channels,
                 out_channels,
                 norm_cfg=dict(type='BN1d', eps=1e-3, momentum=0.01)):
        super(DynamicVFELayer, self).__init__()
        self.fp16_enabled = False
        self.norm = build_norm_layer(norm_cfg, out_channels)[1]
        self.linear = nn.Linear(in_channels, out_channels, bias=False)

    @autocast(dtype=torch.float16)  # ✅ UPDATED: Modern autocast
    def forward(self, inputs):
        """
        Forward function.

        Args:
            inputs (torch.Tensor): Voxels features of shape (M, C).
                M is the number of points, C is the number of channels of point features.

        Returns:
            torch.Tensor: Point features in shape (M, C).
        """
        # [M, in_channels] -> [M, out_channels]
        x = self.linear(inputs)
        x = self.norm(x)
        pointwise = F.relu(x)
        return pointwise


class PFNLayer(nn.Module):
    """
    Pillar Feature Net Layer.
    
    Similar to VFELayer but optimized for pillar-based representations
    commonly used in pillar-based 3D object detection.

    Args:
        in_channels (int): Number of input channels.
        out_channels (int): Number of output channels.
        norm_cfg (dict): Config dict of normalization layers.
        last_layer (bool): Whether this is the last layer.
        mode (str): Pooling mode, either 'max' or 'avg'.
    """

    def __init__(self,
                 in_channels,
                 out_channels,
                 norm_cfg=dict(type='BN1d', eps=1e-3, momentum=0.01),
                 last_layer=False,
                 mode='max'):
        super(PFNLayer, self).__init__()
        self.fp16_enabled = False
        self.last_vfe = last_layer
        self.mode = mode
        
        if not self.last_vfe:
            out_channels = out_channels // 2
        self.units = out_channels

        self.norm = build_norm_layer(norm_cfg, self.units)[1]
        self.linear = nn.Linear(in_channels, self.units, bias=False)

    @autocast(dtype=torch.float16)  # ✅ UPDATED: Modern autocast
    def forward(self, inputs, num_voxels=None):
        """
        Forward function.

        Args:
            inputs (torch.Tensor): Pillar features of shape (N, M, C).
            num_voxels (torch.Tensor, optional): Number of points in each pillar.

        Returns:
            torch.Tensor: Pillar features after encoding.
        """
        x = self.linear(inputs)
        x = self.norm(x.permute(0, 2, 1).contiguous()).permute(0, 2, 1).contiguous()
        x = F.relu(x)

        if self.mode == 'max':
            if self.last_vfe:
                return torch.max(x, dim=1, keepdim=False)[0]
            else:
                x_max = torch.max(x, dim=1, keepdim=True)[0]
                x_repeat = x_max.repeat(1, inputs.shape[1], 1)
                x_concatenated = torch.cat([x, x_repeat], dim=2)
                return x_concatenated
        elif self.mode == 'avg':
            if self.last_vfe:
                return torch.mean(x, dim=1, keepdim=False)
            else:
                x_mean = torch.mean(x, dim=1, keepdim=True)
                x_repeat = x_mean.repeat(1, inputs.shape[1], 1)
                x_concatenated = torch.cat([x, x_repeat], dim=2)
                return x_concatenated


# Test code
if __name__ == "__main__":
    print("Testing VFE layers...")
    
    # Test VFELayer
    print("\n1. Testing VFELayer:")
    vfe = VFELayer(in_channels=4, out_channels=64)
    dummy_input = torch.randn(100, 35, 4)  # [N_voxels, M_points, C_features]
    output = vfe(dummy_input)
    print(f"   Input shape: {dummy_input.shape}")
    print(f"   Output shape: {output.shape}")
    print(f"   ✓ VFELayer works")
    
    # Test DynamicVFELayer
    print("\n2. Testing DynamicVFELayer:")
    dynamic_vfe = DynamicVFELayer(in_channels=4, out_channels=64)
    dummy_input = torch.randn(1000, 4)  # [M_points, C_features]
    output = dynamic_vfe(dummy_input)
    print(f"   Input shape: {dummy_input.shape}")
    print(f"   Output shape: {output.shape}")
    print(f"   ✓ DynamicVFELayer works")
    
    # Test get_paddings_indicator
    print("\n3. Testing get_paddings_indicator:")
    actual_num = torch.tensor([3, 5, 2, 4])  # Number of valid points per voxel
    max_num = 10
    mask = get_paddings_indicator(actual_num, max_num)
    print(f"   Actual nums: {actual_num.tolist()}")
    print(f"   Max num: {max_num}")
    print(f"   Mask shape: {mask.shape}")
    print(f"   ✓ get_paddings_indicator works")
    
    print("\n✅ All tests passed!")
