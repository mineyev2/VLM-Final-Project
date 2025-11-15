# test_registry.py
from mmdet3d.registry import MODELS

# ✅ CRITICAL: Import mmdet3d models first to populate registry
import mmdet3d.models  # This triggers registration

print("="*70)
print("Voxel Encoders in mmdet3d:")
print("="*70)
for key in MODELS.module_dict.keys():
    if 'VFE' in key or 'Voxel' in key.lower():
        print(f"  ✓ {key}")

print("\n" + "="*70)
print("Middle Encoders in mmdet3d:")
print("="*70)
for key in MODELS.module_dict.keys():
    if 'SST' in key or 'Input' in key or 'Middle' in key:
        print(f"  ✓ {key}")

print("\n" + "="*70)
print("Backbones in mmdet3d:")
print("="*70)
for key in MODELS.module_dict.keys():
    if 'SST' in key and 'Input' not in key:
        print(f"  ✓ {key}")

print("\n" + "="*70)
print("All registered models (showing first 20):")
print("="*70)
for i, key in enumerate(sorted(MODELS.module_dict.keys())[:20]):
    print(f"  {i+1}. {key}")

