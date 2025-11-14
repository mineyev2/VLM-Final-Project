"""
Custom operations for SST (window-based operations)
These functions handle voxel-to-window transformations

Version: No torch_scatter dependency (pure PyTorch implementation)
"""

import torch
import torch.nn as nn
import random
import numpy as np
from mmcv.cnn import build_norm_layer
import traceback


def scatter_nd(indices, updates, shape):
    """pytorch edition of tensorflow scatter_nd."""
    ret = torch.zeros(*shape, dtype=updates.dtype, device=updates.device)
    ndim = indices.shape[-1]
    output_shape = list(indices.shape[:-1]) + shape[indices.shape[-1]:]
    flatted_indices = indices.view(-1, ndim)
    slices = [flatted_indices[:, i] for i in range(ndim)]
    slices += [Ellipsis]
    ret[slices] = updates.view(*output_shape)
    return ret


@torch.no_grad()
def get_flat2win_inds(batch_win_inds, voxel_drop_lvl, drop_info, debug=True):
    '''
    Args:
        batch_win_inds: shape=[N, ]. Indicates which window a voxel belongs to.
        voxel_drop_lvl: shape=[N, ]. Indicates batching_level of the window.
    Returns:
        flat2window_inds_dict: contains flat2window_inds of each voxel, shape=[N,]
    '''
    device = batch_win_inds.device
    flat2window_inds_dict = {}

    for dl in drop_info:
        dl_mask = voxel_drop_lvl == dl
        if not dl_mask.any():
            continue

        conti_win_inds = make_continuous_inds(batch_win_inds[dl_mask])
        max_tokens = drop_info[dl]['max_tokens']
        inner_win_inds = get_inner_win_inds(conti_win_inds)
        flat2window_inds = conti_win_inds * max_tokens + inner_win_inds
        flat2window_inds_dict[dl] = (flat2window_inds, torch.where(dl_mask))

        if debug:
            num_windows = len(torch.unique(conti_win_inds))
            assert inner_win_inds.max() < max_tokens
            assert (flat2window_inds >= 0).all()

    return flat2window_inds_dict


def flat2window(feat, voxel_drop_lvl, flat2win_inds_dict, drop_info, padding=0):
    '''
    Args:
        feat: shape=[N, C], N is the voxel num in the batch.
        voxel_drop_lvl: shape=[N, ]. Indicates drop_level of the window.
    Returns:
        feat_3d_dict: contains feat_3d of each drop level. 
            Shape of feat_3d is [num_windows, num_max_tokens, C].
    '''
    dtype = feat.dtype
    device = feat.device
    feat_dim = feat.shape[-1]
    feat_3d_dict = {}

    for dl in drop_info:
        dl_mask = voxel_drop_lvl == dl
        if not dl_mask.any():
            continue

        feat_this_dl = feat[dl_mask]
        this_inds = flat2win_inds_dict[dl][0]
        max_tokens = drop_info[dl]['max_tokens']
        num_windows = (this_inds // max_tokens).max().item() + 1
        
        padding_val = torch.tensor(padding, dtype=dtype, device=device)
        feat_3d = torch.ones((num_windows * max_tokens, feat_dim), dtype=dtype, device=device) * padding_val
        feat_3d[this_inds] = feat_this_dl
        feat_3d = feat_3d.reshape((num_windows, max_tokens, feat_dim))
        feat_3d_dict[dl] = feat_3d

    return feat_3d_dict


def window2flat(feat_3d_dict, inds_dict):
    '''Convert windowed features back to flat representation.'''
    num_all_voxel = 0
    for dl in inds_dict:
        num_all_voxel += inds_dict[dl][0].shape[0]
    
    dtype = feat_3d_dict[list(feat_3d_dict.keys())[0]].dtype
    device = feat_3d_dict[list(feat_3d_dict.keys())[0]].device
    feat_dim = feat_3d_dict[list(feat_3d_dict.keys())[0]].shape[-1]

    all_flat_feat = torch.zeros((num_all_voxel, feat_dim), device=device, dtype=dtype)

    for dl in feat_3d_dict:
        feat = feat_3d_dict[dl]
        feat_dim = feat.shape[-1]
        inds, flat_pos = inds_dict[dl]
        feat = feat.reshape(-1, feat_dim)
        flat_feat = feat[inds]
        all_flat_feat[flat_pos] = flat_feat
    
    return all_flat_feat


def get_flat2win_inds_v2(batch_win_inds, voxel_drop_lvl, drop_info, debug=True):
    '''V2 wrapper that includes metadata.'''
    transform_dict = get_flat2win_inds(batch_win_inds, voxel_drop_lvl, drop_info, debug)
    transform_dict['voxel_drop_level'] = voxel_drop_lvl
    transform_dict['batching_info'] = drop_info
    return transform_dict


def window2flat_v2(feat_3d_dict, inds_dict):
    '''V2 wrapper for window2flat.'''
    inds_v1 = {k: inds_dict[k] for k in inds_dict if not isinstance(k, str)}
    return window2flat(feat_3d_dict, inds_v1)


def flat2window_v2(feat, inds_dict, padding=0):
    '''V2 wrapper for flat2window.'''
    assert 'voxel_drop_level' in inds_dict, 'voxel_drop_level should be in inds_dict'
    inds_v1 = {k: inds_dict[k] for k in inds_dict if not isinstance(k, str)}
    batching_info = inds_dict['batching_info']
    return flat2window(feat, inds_dict['voxel_drop_level'], inds_v1, batching_info, padding=padding)


@torch.no_grad()
def get_inner_win_inds(win_inds):
    '''
    Get inner window indices for voxels (pure PyTorch implementation).
    
    Args:
        win_inds: shape=[N,]. Window indices for each voxel.
    Returns:
        inner_inds: shape=[N,]. Position of each voxel within its window.
    '''
    # Sort by window index
    sorted_inds, order = win_inds.sort()
    
    # Find where windows change
    roll_inds = torch.roll(sorted_inds, -1)
    diff = sorted_inds - roll_inds
    end_pos_mask = diff != 0
    
    # Count voxels per window
    bincount = torch.bincount(win_inds)
    unique_inds = torch.unique(win_inds)
    num_tokens_each_win = bincount[unique_inds]
    
    # Build inner indices
    template = torch.ones_like(win_inds)
    template[end_pos_mask] = (num_tokens_each_win - 1) * -1
    inner_inds = torch.cumsum(template, 0)
    inner_inds[end_pos_mask] = num_tokens_each_win
    inner_inds -= 1
    
    # Recover original order
    inner_inds_reorder = torch.empty_like(win_inds)
    inner_inds_reorder[order] = inner_inds
    
    return inner_inds_reorder


@torch.no_grad()
def get_window_coors(coors, sparse_shape, window_shape, do_shift):
    '''
    Compute window coordinates and positions within windows.
    
    Args:
        coors: [N, 4] voxel coordinates (batch, z, y, x)
        sparse_shape: tuple (X, Y, Z) spatial dimensions
        window_shape: tuple (win_x, win_y) or (win_x, win_y, win_z)
        do_shift: whether to apply window shifting
        
    Returns:
        batch_win_inds: [N] window index for each voxel
        coors_in_win: [N, 3] position within window (z, y, x)
    '''
    if len(window_shape) == 2:
        win_shape_x, win_shape_y = window_shape
        win_shape_z = sparse_shape[-1]
    else:
        win_shape_x, win_shape_y, win_shape_z = window_shape

    sparse_shape_x, sparse_shape_y, sparse_shape_z = sparse_shape

    max_num_win_x = int(np.ceil((sparse_shape_x / win_shape_x)) + 1)
    max_num_win_y = int(np.ceil((sparse_shape_y / win_shape_y)) + 1)
    max_num_win_z = int(np.ceil((sparse_shape_z / win_shape_z)) + 1)
    max_num_win_per_sample = max_num_win_x * max_num_win_y * max_num_win_z

    if do_shift:
        shift_x = win_shape_x // 2
        shift_y = win_shape_y // 2
        shift_z = win_shape_z // 2
    else:
        shift_x = shift_y = shift_z = 0
    
    # Compatibility for 2D windows
    if sparse_shape_z == win_shape_z:
        shift_z = 0

    # Apply shifts
    shifted_coors_x = coors[:, 3] + shift_x
    shifted_coors_y = coors[:, 2] + shift_y
    shifted_coors_z = coors[:, 1] + shift_z

    # Compute window coordinates
    win_coors_x = shifted_coors_x // win_shape_x
    win_coors_y = shifted_coors_y // win_shape_y
    win_coors_z = shifted_coors_z // win_shape_z

    # Compute global window index
    batch_win_inds = (coors[:, 0] * max_num_win_per_sample +
                     win_coors_x * max_num_win_y * max_num_win_z +
                     win_coors_y * max_num_win_z +
                     win_coors_z)

    # Compute position within window
    coors_in_win_x = shifted_coors_x % win_shape_x
    coors_in_win_y = shifted_coors_y % win_shape_y
    coors_in_win_z = shifted_coors_z % win_shape_z
    coors_in_win = torch.stack([coors_in_win_z, coors_in_win_y, coors_in_win_x], dim=-1)
    
    return batch_win_inds, coors_in_win


@torch.no_grad()
def make_continuous_inds(inds):
    '''Make indices continuous starting from 0.'''
    dtype = inds.dtype
    device = inds.device

    unique_inds = torch.sort(torch.unique(inds))[0]
    num_valid_inds = len(unique_inds)
    max_origin_inds = unique_inds.max().item()
    
    canvas = torch.full((max_origin_inds + 1,), -1, dtype=dtype, device=device)
    canvas[unique_inds] = torch.arange(num_valid_inds, dtype=dtype, device=device)
    
    conti_inds = canvas[inds]
    return conti_inds


def scatter_v2(feat, coors, mode, return_inv=True, min_points=0, unq_inv=None, new_coors=None):
    '''
    Scatter features to unique coordinates.
    Pure PyTorch implementation (no torch_scatter dependency).
    '''
    assert feat.size(0) == coors.size(0)
    if mode == 'avg':
        mode = 'mean'

    # Get unique coordinates
    if unq_inv is None and min_points > 0:
        new_coors, unq_inv, unq_cnt = torch.unique(coors, return_inverse=True, return_counts=True, dim=0)
    elif unq_inv is None:
        new_coors, unq_inv = torch.unique(coors, return_inverse=True, return_counts=False, dim=0)
        unq_cnt = None
    else:
        assert new_coors is not None
        unq_cnt = None

    # Filter by minimum points if needed
    if min_points > 0:
        if unq_cnt is None:
            _, unq_cnt = torch.unique(unq_inv, return_counts=True)
        cnt_per_point = unq_cnt[unq_inv]
        valid_mask = cnt_per_point >= min_points
        feat = feat[valid_mask]
        coors = coors[valid_mask]
        new_coors, unq_inv, unq_cnt = torch.unique(coors, return_inverse=True, return_counts=True, dim=0)

    # Pure PyTorch scatter operations
    num_unique = new_coors.size(0)
    feat_dim = feat.size(1) if len(feat.shape) > 1 else 1
    
    if mode == 'max':
        # Initialize with very negative values
        new_feat = torch.full((num_unique, feat_dim), float('-inf'), 
                             dtype=feat.dtype, device=feat.device)
        # Scatter max using index_reduce (PyTorch 1.12+) or manual implementation
        if hasattr(torch, 'index_reduce'):
            new_feat = torch.index_reduce(new_feat, 0, unq_inv, feat, 'amax', include_self=False)
        else:
            # Manual max scatter
            for i in range(num_unique):
                mask = unq_inv == i
                if mask.any():
                    new_feat[i] = feat[mask].max(dim=0)[0]
        argmax = None  # Not computing argmax in pure PyTorch version
        
    elif mode == 'mean':
        # Use scatter_add and divide by counts
        new_feat = torch.zeros((num_unique, feat_dim), dtype=feat.dtype, device=feat.device)
        new_feat.scatter_add_(0, unq_inv.unsqueeze(1).expand_as(feat), feat)
        if unq_cnt is None:
            _, unq_cnt = torch.unique(unq_inv, return_counts=True)
        new_feat = new_feat / unq_cnt.unsqueeze(1).float()
        
    elif mode == 'sum':
        # Use scatter_add
        new_feat = torch.zeros((num_unique, feat_dim), dtype=feat.dtype, device=feat.device)
        new_feat.scatter_add_(0, unq_inv.unsqueeze(1).expand_as(feat), feat)
    else:
        raise NotImplementedError(f"Mode {mode} not implemented")

    if not return_inv:
        return new_feat, new_coors
    else:
        return new_feat, new_coors, unq_inv


# Additional utility functions
def build_mlp(in_channel, hidden_dims, norm_cfg, is_head=False, act='relu', bias=False, dropout=0):
    '''Build MLP layers.'''
    layer_list = []
    last_channel = in_channel
    if isinstance(hidden_dims, int):
        hidden_dims = [hidden_dims]
    
    for i, c in enumerate(hidden_dims):
        act_layer = get_activation_layer(act, c)
        norm_layer = build_norm_layer(norm_cfg, c)[1]
        
        if i == len(hidden_dims) - 1 and is_head:
            layer_list.append(nn.Linear(last_channel, c, bias=True))
        else:
            sq = [
                nn.Linear(last_channel, c, bias=bias),
                norm_layer,
                act_layer,
            ]
            if dropout > 0:
                sq.append(nn.Dropout(dropout))
            layer_list.append(nn.Sequential(*sq))
        
        last_channel = c
    
    mlp = nn.Sequential(*layer_list)
    return mlp


def get_activation_layer(act, dim=None):
    '''Get activation layer by name.'''
    act = act.lower()
    if act == 'relu':
        return nn.ReLU(inplace=True)
    elif act == 'gelu':
        return nn.GELU()
    elif act == 'leakyrelu':
        return nn.LeakyReLU(inplace=True)
    elif act == 'prelu':
        return nn.PReLU(num_parameters=dim)
    elif act in ('swish', 'silu'):
        return nn.SiLU(inplace=True)
    elif act == 'glu':
        return nn.GLU()
    elif act == 'elu':
        return nn.ELU(inplace=True)
    else:
        raise NotImplementedError(f"Activation {act} not implemented")


# Test code
if __name__ == "__main__":
    print("Testing sst_ops (no torch_scatter)...")
    print("✅ All imports successful - pure PyTorch implementation!")
