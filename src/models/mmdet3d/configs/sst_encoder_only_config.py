# SST Encoder-Only Configuration
# Extracted from your full detection config

model = dict(
    type="SSTv2",
    
    # Architecture (from your num_encoder_layers=4)
    d_model=[128, 128, 128, 128],
    nhead=[8, 8, 8, 8],
    num_blocks=4,
    dim_feedforward=[256, 256, 256, 256],
    
    # Activation and regularization
    dropout=0.0,
    activation="gelu",
    
    # Output configuration (from your backbone)
    output_shape=(80, 80),
    
    # Convolution layers (from your backbone)
    num_attached_conv=0,
    conv_in_channel=128,
    conv_out_channel=128,
    conv_kwargs=[
        dict(kernel_size=3, dilation=1, padding=1, stride=1),
        dict(kernel_size=3, dilation=1, padding=1, stride=1),
        dict(kernel_size=3, dilation=2, padding=2, stride=1),
    ],
    
    # Normalization config (updated for modern mmcv)
    norm_cfg=dict(type='BN2d', eps=1e-3, momentum=0.01),
    conv_cfg=dict(type='Conv2d', bias=False),
    
    # Input channel
    in_channel=128,
    
    # Optional features
    checkpoint_blocks=[],
    layer_cfg=dict(),
    debug=True,
)

# These are not used for encoder-only, but kept for compatibility
train_cfg = None
test_cfg = None
