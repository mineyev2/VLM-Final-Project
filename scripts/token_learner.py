import torch
import torch.nn as nn
import torch.nn.functional as F

class TokenLearner(nn.Module):
    def __init__(self, in_channels, num_tokens):
        """
        Args:
            in_channels (int): The number of channels 'C' in the input.
            num_tokens (int): The target number of tokens (e.g., 16).
        """
        super().__init__()
        self.num_tokens = num_tokens

        # Implemented with four 3x3 conv layers with gelu in between. Channel size is identical to num_tokens.
        self.attention_network = nn.Sequential(
            nn.Conv2d(in_channels, num_tokens, kernel_size=3, padding=1),
            nn.GELU(),
            nn.Conv2d(num_tokens, num_tokens, kernel_size=3, padding=1),
            nn.GELU(),
            nn.Conv2d(num_tokens, num_tokens, kernel_size=3, padding=1),
            nn.GELU(),
            nn.Conv2d(num_tokens, num_tokens, kernel_size=3, padding=1),
            # Ends with a Sigmoid as per Eq (1) 
            nn.Sigmoid() 
        )

    def forward(self, x):
        """
        Args:
            x: Input tensor of shape (Batch, Channels, Height, Width)
        Returns:
            Output tensor of shape (Batch, num_tokens, Channels)
        """
        
        attention_maps = self.attention_network(x)

        # Apply Attention and Spatial Pooling
        # We want to perform the element-wise multiplication and global average pooling.
        # Equation: z_i = rho(X_t * alpha_i(X_t)) 
        
        # We can use Einstein Summation for clean, efficient multiplication + summing.
        # 'bchw' = input (Batch, Channel, Height, Width)
        # 'bshw' = attention_maps (Batch, Tokens, Height, Width)
        # 'bsc'  = output (Batch, Tokens, Channel)
        
        # Note: The paper mentions spatial global average pooling. 
        # Einsum sums over H and W. To get the average, we divide by H*W.
        feature_sum = torch.einsum('bchw,bshw->bsc', x, attention_maps)
        
        N = x.shape[2] * x.shape[3] # Height * Width
        tokens = feature_sum / N
        
        return tokens
