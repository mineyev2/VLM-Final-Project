"""
Dynamic Scatter Operation for SST
Converts point clouds to voxels dynamically (no pre-allocated grid)

This module provides both forward and backward passes for dynamic voxelization.
Requires compiled CUDA extension: voxel_layer
"""

import torch
from torch import nn
from torch.autograd import Function

# Import CUDA operations
# These are compiled from C++/CUDA source files
try:
    from .voxel_layer import (
        dynamic_point_to_voxel_forward,
        dynamic_point_to_voxel_backward
    )
    VOXEL_LAYER_AVAILABLE = True
except ImportError:
    # Fallback message if CUDA ops not available
    VOXEL_LAYER_AVAILABLE = False
    print("Warning: voxel_layer CUDA extension not found. DynamicScatter will fail at runtime.")


class _DynamicScatter(Function):
    """
    Autograd function for dynamic scatter operation.
    
    Scatters point features into voxels using max/mean/sum reduction.
    """

    @staticmethod
    def forward(ctx, feats, coors, reduce_type='max'):
        """Convert point features to voxel features.

        Args:
            feats (Tensor): [N, C] float tensor. Point features to be reduced into voxels.
            coors (Tensor): [N, ndim] int tensor. Voxel coordinates of each point.
            reduce_type (str): Reduction operation. Options: 'max', 'sum', 'mean'
            
        Returns:
            tuple:
                voxel_feats (Tensor): [M, C] reduced voxel features
                voxel_coors (Tensor): [M, ndim] voxel coordinates
        """
        if not VOXEL_LAYER_AVAILABLE:
            raise ImportError(
                "voxel_layer CUDA extension not available. "
                "Please compile the extension or check mmdet3d installation."
            )
        
        # Call CUDA forward operation
        results = dynamic_point_to_voxel_forward(feats, coors, reduce_type)
        voxel_feats, voxel_coors, point2voxel_map, voxel_points_count = results
        
        # Save for backward pass
        ctx.reduce_type = reduce_type
        ctx.save_for_backward(feats, voxel_feats, point2voxel_map, voxel_points_count)
        ctx.mark_non_differentiable(voxel_coors)
        
        return voxel_feats, voxel_coors

    @staticmethod
    def backward(ctx, grad_voxel_feats, grad_voxel_coors=None):
        """Backward pass for dynamic scatter.
        
        Args:
            grad_voxel_feats (Tensor): Gradient w.r.t. voxel features
            grad_voxel_coors (Tensor, optional): Gradient w.r.t. coordinates (unused)
            
        Returns:
            tuple: (grad_feats, None, None) - gradient w.r.t. point features
        """
        feats, voxel_feats, point2voxel_map, voxel_points_count = ctx.saved_tensors
        
        # Initialize gradient
        grad_feats = torch.zeros_like(feats)
        
        # Call CUDA backward operation
        dynamic_point_to_voxel_backward(
            grad_feats,
            grad_voxel_feats.contiguous(),
            feats,
            voxel_feats,
            point2voxel_map,
            voxel_points_count,
            ctx.reduce_type
        )
        
        return grad_feats, None, None


# Expose the function
dynamic_scatter = _DynamicScatter.apply


class DynamicScatter(nn.Module):
    """
    Dynamic voxelization module.
    
    Scatters points into voxels without pre-allocating a fixed grid.
    Automatically handles batched inputs.
    
    Args:
        voxel_size (list[float]): Size of each voxel [x, y, z]
        point_cloud_range (list[float]): Range of point cloud [x_min, y_min, z_min, x_max, y_max, z_max]
        average_points (bool): If True, use mean pooling; otherwise use max pooling
        
    Note:
        CPU and GPU implementations may have small numerical differences (~5e-7)
        after summation and division operations.
    """

    def __init__(self, voxel_size, point_cloud_range, average_points):
        super(DynamicScatter, self).__init__()
        self.voxel_size = voxel_size
        self.point_cloud_range = point_cloud_range
        self.average_points = average_points

    def forward_single(self, points, coors):
        """Process a single sample (no batch dimension).
        
        Args:
            points (Tensor): [N, C] point features
            coors (Tensor): [N, 3] voxel coordinates (z, y, x)
            
        Returns:
            tuple: (voxel_feats, voxel_coors)
        """
        reduce = 'mean' if self.average_points else 'max'
        return dynamic_scatter(points.contiguous(), coors.contiguous(), reduce)

    def forward(self, points, coors):
        """
        Forward pass supporting both single and batched inputs.
        
        Args:
            points (Tensor): [N, C] point features
            coors (Tensor): [N, 3] or [N, 4] voxel coordinates
                            If 4D: [batch_idx, z, y, x]
                            If 3D: [z, y, x] (single sample)
                            
        Returns:
            tuple:
                features (Tensor): [M, C] voxel features
                feature_coors (Tensor): [M, 3] or [M, 4] voxel coordinates
        """
        # Single sample case (3D coordinates)
        if coors.size(-1) == 3:
            return self.forward_single(points, coors)
        
        # Batched case (4D coordinates with batch index)
        batch_size = coors[-1, 0].int().item() + 1
        voxels, voxel_coors = [], []
        
        for i in range(batch_size):
            # Get points for this batch
            batch_mask = (coors[:, 0] == i)
            batch_points = points[batch_mask]
            batch_coors = coors[batch_mask][:, 1:]  # Remove batch dimension
            
            # Process single batch
            voxel, voxel_coor = self.forward_single(batch_points, batch_coors)
            
            # Add batch index back
            coor_pad = nn.functional.pad(
                voxel_coor, (1, 0), mode='constant', value=i
            )
            voxel_coors.append(coor_pad)
            voxels.append(voxel)
        
        # Concatenate all batches
        features = torch.cat(voxels, dim=0)
        feature_coors = torch.cat(voxel_coors, dim=0)
        
        return features, feature_coors

    def __repr__(self):
        """String representation of the module."""
        return (
            f'{self.__class__.__name__}('
            f'voxel_size={self.voxel_size}, '
            f'point_cloud_range={self.point_cloud_range}, '
            f'average_points={self.average_points})'
        )


# For backward compatibility
__all__ = ['DynamicScatter', 'dynamic_scatter']
