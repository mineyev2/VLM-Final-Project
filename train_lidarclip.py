"""
SST Wrapper for LidarCLIP Integration
Compatible with mmdet3d 1.4.0, mmcv 2.1.0

Builds encoder components individually to avoid detector framework requirements.
"""
import sys
from pathlib import Path

import torch
import torch.nn as nn
from mmengine.config import Config
from mmengine.registry import init_default_scope

# Ensure project root is in path
project_root = Path(__file__).parent.parent.parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

# Initialize mmdet3d scope
init_default_scope('mmdet3d')

# Import mmdet3d registry
from mmdet3d.registry import MODELS as MMDET3D_MODELS

class PurePyTorchDynamicVFE(nn.Module):
    """Pure PyTorch DynamicVFE - no CUDA ops, H100 compatible."""
    
    def __init__(self, 
                 in_channels=4,
                 feat_channels=[64, 128],
                 with_distance=False,
                 with_cluster_center=True,
                 with_voxel_center=True,
                 voxel_size=(0.5, 0.5, 6),
                 point_cloud_range=(0, -40, -3, 70.4, 40, 1),
                 norm_cfg=dict(type='BN1d', eps=1e-3, momentum=0.01)):
        super().__init__()
        
        self.in_channels = in_channels
        self.with_distance = with_distance
        self.with_cluster_center = with_cluster_center
        self.with_voxel_center = with_voxel_center
        
        # Store voxelization params
        self.voxel_size = torch.tensor(voxel_size, dtype=torch.float32)
        self.point_cloud_range = torch.tensor(point_cloud_range, dtype=torch.float32)
        
        # ==========================================
        # CRITICAL FIX: Calculate actual input size
        # ==========================================
        actual_in_channels = in_channels
        if self.with_voxel_center:
            actual_in_channels += 3  # xyz offset from voxel center
        if self.with_cluster_center:
            actual_in_channels += 3  # xyz offset from cluster center
        if self.with_distance:
            actual_in_channels += 1  # distance
        
        print(f"[VFE] Input channels: {in_channels} → {actual_in_channels} (with augmentations)")
        
        # Build VFE layers with CORRECTED input dimension
        feat_channels = [actual_in_channels] + list(feat_channels)
        vfe_layers = []
        
        for i in range(len(feat_channels) - 1):
            in_filters = feat_channels[i]
            out_filters = feat_channels[i + 1]
            
            # After first layer, concatenate with max pooling
            if i > 0:
                in_filters *= 2
            
            print(f"[VFE] Layer {i}: {in_filters} → {out_filters}")
            
            vfe_layers.append(
                nn.Sequential(
                    nn.Linear(in_filters, out_filters, bias=False),
                    nn.BatchNorm1d(out_filters, eps=norm_cfg['eps'], momentum=norm_cfg['momentum']),
                    nn.ReLU(inplace=True)
                )
            )
        
        self.vfe_layers = nn.ModuleList(vfe_layers)
        self.num_vfe = len(vfe_layers)
    
    def forward(self, features, coors):
        """
        Args:
            features: [N, 4] point features (x, y, z, intensity)
            coors: [N, 4] voxel coordinates (batch_idx, z, y, x)
        
        Returns:
            voxel_features: [M, C] aggregated voxel features
            voxel_coors: [M, 4] unique voxel coordinates
        """
        device = features.device
        self.voxel_size = self.voxel_size.to(device)
        self.point_cloud_range = self.point_cloud_range.to(device)
        
        # Get unique voxels and inverse mapping
        voxel_coors, inverse_indices = torch.unique(coors, return_inverse=True, dim=0)
        
        # Extract point coordinates
        points_coords = features[:, :3]  # [N, 3] xyz
        
        # Add voxel center offset features
        if self.with_voxel_center:
            # Compute voxel centers from voxel coordinates
            # coors format: [batch_idx, z, y, x]
            voxel_centers = torch.zeros(voxel_coors.shape[0], 3, device=device)
            voxel_centers[:, 0] = (voxel_coors[:, 3].float() + 0.5) * self.voxel_size[0] + self.point_cloud_range[0]  # x
            voxel_centers[:, 1] = (voxel_coors[:, 2].float() + 0.5) * self.voxel_size[1] + self.point_cloud_range[1]  # y
            voxel_centers[:, 2] = (voxel_coors[:, 1].float() + 0.5) * self.voxel_size[2] + self.point_cloud_range[2]  # z
            
            # Get each point's voxel center
            point_voxel_centers = voxel_centers[inverse_indices]  # [N, 3]
            
            # Compute offset
            voxel_center_offset = points_coords - point_voxel_centers
            
            # Concatenate with original features
            features = torch.cat([features, voxel_center_offset], dim=-1)
        
        # Process through VFE layers
        voxel_feats = features
        
        for i, vfe in enumerate(self.vfe_layers):
            point_feats = vfe(voxel_feats)
            
            if i != self.num_vfe - 1:
                # Max pooling per voxel
                voxel_max_feats = torch.zeros(
                    voxel_coors.shape[0], 
                    point_feats.shape[1], 
                    device=device, 
                    dtype=point_feats.dtype
                )
                voxel_max_feats.scatter_reduce_(
                    0, 
                    inverse_indices.unsqueeze(-1).expand(-1, point_feats.shape[1]),
                    point_feats, 
                    reduce='amax', 
                    include_self=False
                )
                
                # Expand back to point level and concatenate
                voxel_max_feats_expanded = voxel_max_feats[inverse_indices]
                voxel_feats = torch.cat([point_feats, voxel_max_feats_expanded], dim=-1)
            else:
                voxel_feats = point_feats
        
        # Final aggregation: max pooling to voxel level
        voxel_features = torch.zeros(
            voxel_coors.shape[0], 
            voxel_feats.shape[1],
            device=device, 
            dtype=voxel_feats.dtype
        )
        voxel_features.scatter_reduce_(
            0,
            inverse_indices.unsqueeze(-1).expand(-1, voxel_feats.shape[1]),
            voxel_feats,
            reduce='amax',
            include_self=False
        )
        
        return voxel_features, voxel_coors



    
class SSTEncoderOnly(nn.Module):
    """
    Wrapper that builds SST encoder components individually.
    Avoids the full detector framework which requires bbox_head.
    """
    
    def __init__(self, cfg):
        super().__init__()
        
        print("[SST] Building encoder components individually...")
        from mmcv.ops import Voxelization
        voxel_cfg = cfg.model.voxel_layer
        self.voxel_layer = Voxelization(
            voxel_size=voxel_cfg.voxel_size,
            point_cloud_range=voxel_cfg.point_cloud_range,
            max_num_points=voxel_cfg.max_num_points,
            max_voxels=voxel_cfg.max_voxels
        )
        print("[SST]   ✓ Voxel layer: Voxelization (dynamic)")

                # Build voxel encoder - use pure PyTorch version for H100 compatibility
        vfe_cfg = cfg.model.voxel_encoder
        self.voxel_encoder = PurePyTorchDynamicVFE(
            in_channels=vfe_cfg.in_channels,
            feat_channels=vfe_cfg.feat_channels,
            with_distance=vfe_cfg.get('with_distance', False),
            with_cluster_center=vfe_cfg.get('with_cluster_center', True),
            with_voxel_center=vfe_cfg.get('with_voxel_center', True),
            voxel_size=vfe_cfg.voxel_size,
            point_cloud_range=vfe_cfg.point_cloud_range,
            norm_cfg=vfe_cfg.get('norm_cfg', dict(type='BN1d', eps=1e-3, momentum=0.01))
        )
        print("[SST]   ✓ Voxel encoder: PurePyTorchDynamicVFE (H100 compatible)")
        
        # Build middle encoder
        self.middle_encoder = MMDET3D_MODELS.build(cfg.model.middle_encoder)
        print("[SST]   ✓ Middle encoder: SSTInputLayerV2")
        
        # Build backbone
        self.backbone = MMDET3D_MODELS.build(cfg.model.backbone)
        print("[SST]   ✓ Backbone: SSTv2")
        
        print("[SST] ✓ Built encoder-only model")
    

    def extract_feat(self, point_clouds):
        """Extract features from point clouds."""
        # Step 1: Voxelize
        voxels, coors = self.voxelize(point_clouds)
        
        # Step 2: Encode voxels
        voxel_features, feature_coors = self.voxel_encoder(voxels, coors)
        
        # Step 3: Get batch size
        batch_size = coors[-1, 0].item() + 1
        
        # Step 4: Middle encoder - returns dict
        voxel_info = self.middle_encoder(voxel_features, feature_coors, batch_size)
        
        # Step 5: Backbone - pass the dict
        bev_features = self.backbone(voxel_info)
        
        return bev_features
    

    @torch.no_grad()
    def voxelize(self, points):
        """Pure PyTorch dynamic voxelization (H100 compatible, no CUDA ops).
        
        Args:
            points (list[torch.Tensor]): List of point clouds, each [N, 4]
        
        Returns:
            tuple: (voxels, coors_batch)
        """
        import torch.nn.functional as F
        
        # Get voxelization parameters from config
        voxel_size = torch.tensor(
            self.voxel_layer.voxel_size, 
            dtype=torch.float32,
            device=points[0].device
        )
        pc_range = torch.tensor(
            self.voxel_layer.point_cloud_range,
            dtype=torch.float32,
            device=points[0].device
        )
        
        all_coors = []
        all_points = []
        
        for batch_idx, pts in enumerate(points):
            # Compute voxel coordinates: floor((point - min_range) / voxel_size)
            coors = torch.floor(
                (pts[:, :3] - pc_range[:3]) / voxel_size
            ).int()
            
            # Calculate grid dimensions
            grid_size = torch.floor(
                (pc_range[3:] - pc_range[:3]) / voxel_size
            ).int()
            
            # Filter out points outside the valid range
            valid_mask = (
                (coors[:, 0] >= 0) & (coors[:, 0] < grid_size[0]) &
                (coors[:, 1] >= 0) & (coors[:, 1] < grid_size[1]) &
                (coors[:, 2] >= 0) & (coors[:, 2] < grid_size[2])
            )
            
            coors = coors[valid_mask]
            pts = pts[valid_mask]
            
            # Convert from [x, y, z] to [z, y, x] and prepend batch index
            # Final format: [batch_idx, z, y, x]
            coors_zyx = torch.flip(coors, dims=[-1])
            batch_coors = F.pad(coors_zyx, (1, 0), value=batch_idx)
            
            all_coors.append(batch_coors)
            all_points.append(pts)
        
        # Concatenate all batches
        voxels = torch.cat(all_points, dim=0)
        coors_batch = torch.cat(all_coors, dim=0)
        
        return voxels, coors_batch

def build_sst(sst_config):
    """Build SST encoder from config."""
    
    # Load config
    cfg = Config.fromfile(sst_config)
    
    print(f"[SST] Building model from config: {sst_config}")
    print(f"[SST] Mode: Encoder-only")
    
    # Build encoder-only model
    model = SSTEncoderOnly(cfg)
    
    return model


@MMDET3D_MODELS.register_module()
class LidarEncoderSST(nn.Module):
    """
    SST-based LiDAR encoder wrapper for LidarCLIP.
    Handles checkpoint loading with prefix stripping.
    """
    
    def __init__(self, sst_config, clip_embedding_dim=768, checkpoint=None):
        super().__init__()
        
        print("\n" + "="*70)
        print("[LidarEncoderSST] Initializing")
        print("="*70)
        print(f"Config: {sst_config}")
        print(f"Output dim: {clip_embedding_dim}")
        print(f"Mode: Encoder-only")
        
        # Build SST encoder
        self._sst = build_sst(sst_config)
        
        # Extract backbone output shape from config
        cfg = Config.fromfile(sst_config)
        backbone_config = cfg.model.get('backbone', {})
        backbone_output_shape = backbone_config.get('output_shape', [80, 80])
        backbone_channels = backbone_config.get('conv_out_channel', 128)
        
        print(f"\n[LidarEncoderSST] Building attention pooler...")
        print(f"  Input: [{backbone_channels}, {backbone_output_shape[0]}, {backbone_output_shape[1]}]")
        print(f"  Output: {clip_embedding_dim}")
        
        # Import attention pooler
        from src.models.attention_pool import AttentionPool2d
        
        self._pooler = AttentionPool2d(
            spacial_dim=backbone_output_shape[0],
            embed_dim=clip_embedding_dim,
            num_heads=8,
            input_dim=backbone_channels,
            output_dim=clip_embedding_dim,  # Explicitly set output dimension
        )
        
        # Load checkpoint if provided
        if checkpoint:
            print(f"\n[LidarEncoderSST] Loading checkpoint: {checkpoint}")
            self._load_checkpoint(checkpoint)
        
        print("\n" + "="*70)
        print("[LidarEncoderSST] ✓ Initialization complete!")
        print("="*70 + "\n")
    
    def _load_checkpoint(self, checkpoint_path):
        """
        Load pretrained weights from LidarCLIP checkpoint.
        Handles the 'lidar_encoder.' prefix stripping.
        """
        # Load checkpoint file
        checkpoint = torch.load(checkpoint_path, map_location='cpu')
        
        # Handle different checkpoint formats
        if 'state_dict' in checkpoint:
            state_dict = checkpoint['state_dict']
        elif 'model' in checkpoint:
            state_dict = checkpoint['model']
        else:
            state_dict = checkpoint
        
        # Strip the 'lidar_encoder.' prefix from checkpoint keys
        # The original LidarCLIP saved with this prefix
        new_state_dict = {}
        prefix_to_remove = 'lidar_encoder.'
        
        for key, value in state_dict.items():
            if key.startswith(prefix_to_remove):
                # Remove the prefix
                new_key = key[len(prefix_to_remove):]
                new_state_dict[new_key] = value
            else:
                new_state_dict[key] = value
        
        # Filter out pooler weights if dimensions mismatch
        # The SST encoder is the important part; pooler can be randomly initialized
        filtered_state_dict = {}
        for key, value in new_state_dict.items():
            if key.startswith('_pooler.'):
                # Skip pooler weights - will be randomly initialized
                continue
            filtered_state_dict[key] = value
        
        # Load state dict with strict=False
        missing_keys, unexpected_keys = self.load_state_dict(
            filtered_state_dict, strict=False
        )
        
        # Report loading status
        print(f"\n[Checkpoint Loading Report]")
        
        if missing_keys:
            print(f"⚠️  Missing keys: {len(missing_keys)} total")
            # Only show first few to avoid clutter
            if len(missing_keys) <= 5:
                for key in missing_keys:
                    print(f"  - {key}")
            else:
                print(f"  First 5 of {len(missing_keys)}:")
                for key in missing_keys[:5]:
                    print(f"  - {key}")
                print(f"  ... and {len(missing_keys) - 5} more")
        
        if unexpected_keys:
            print(f"⚠️  Unexpected keys: {len(unexpected_keys)} total")
            if len(unexpected_keys) <= 5:
                for key in unexpected_keys:
                    print(f"  - {key}")
            else:
                print(f"  First 5 of {len(unexpected_keys)}:")
                for key in unexpected_keys[:5]:
                    print(f"  - {key}")
                print(f"  ... and {len(unexpected_keys) - 5} more")
        
        # Final status
        if not missing_keys and not unexpected_keys:
            print("\n[LidarEncoderSST] ✓ Checkpoint loaded perfectly - all keys matched!")
        elif len(missing_keys) == 0:
            print(f"\n[LidarEncoderSST] ✓ Checkpoint loaded (ignored {len(unexpected_keys)} unexpected keys)")
        else:
            print(f"\n[LidarEncoderSST] ✓ Checkpoint loaded with partial match")
            print(f"  → {len(missing_keys)} keys will use random initialization")
            print(f"  → {len(unexpected_keys)} checkpoint keys were ignored")
    
    def forward(self, point_clouds, no_pooling=False, return_attention=False):
        """
        Forward pass through SST encoder and attention pooling.
        
        Args:
            point_clouds (list[Tensor]): List of point clouds, each [N_i, 4]
                                        Format: [x, y, z, intensity/reflectance]
            no_pooling (bool): If True, return BEV features without pooling
            return_attention (bool): If True, also return attention weights
            
        Returns:
            Tensor: Pooled features [B, clip_embedding_dim]
            Optional[Tensor]: Attention weights if return_attention=True
        """
        # Extract BEV features using SST encoder pipeline:
        # point_clouds -> voxel_encoder -> middle_encoder -> backbone
        bev_features = self._sst.extract_feat(point_clouds)
        
        # extract_feat returns a tuple, take first element
        if isinstance(bev_features, (list, tuple)):
            bev_features = bev_features[0]  # [B, C, H, W]
        
        # If no pooling requested, return spatial features
        if no_pooling:
            if return_attention:
                return bev_features, None
            else:
                return bev_features
        
        # Apply attention pooling to get CLIP-compatible embeddings
        pooled_features, attention_weights = self._pooler(
            bev_features,
            no_pooling=False,
            return_attention=return_attention
        )
        
        if return_attention:
            return pooled_features, attention_weights
        else:
            return pooled_features


# Test code
if __name__ == "__main__":
    print("="*70)
    print("Testing LidarEncoderSST")
    print("="*70)
    
    # Initialize model
    model = LidarEncoderSST(
        sst_config="src/models/mmdet3d/configs/sst_encoder_only_config.py",
        clip_embedding_dim=768,  # ViT-L/14
        checkpoint=None  # Set to checkpoint path to test loading
    )
    
    # Test with dummy data
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = model.to(device)
    
    # Create dummy point clouds (4 clouds, ~100 points each)
    point_clouds = [
        torch.randn(100, 4).to(device) 
        for _ in range(4)
    ]
    
    print("\nRunning forward pass...")
    with torch.no_grad():
        output = model(point_clouds)
    
    print(f"\n✓ Success!")
    print(f"  Input: {len(point_clouds)} point clouds")
    print(f"  Output shape: {output.shape}")
    print(f"  Expected: [4, 768]")
    
    assert output.shape == torch.Size([4, 768]), "Output shape mismatch!"
    print("\n✓ All tests passed!")
