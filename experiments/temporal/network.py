import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models

class ResNet50Regressor(nn.Module):
    """
    Base UAV bounding box regressor using a ResNet50 backbone.
    Input: Single image [B, C, H, W]
    Output: Bounding box predictions [B, 4] (normalized cx, cy, w, h)
    """
    def __init__(self, overfit=False, num_img_channels=3):
        super().__init__()
        self.backbone = models.resnet50(
            weights=models.ResNet50_Weights.IMAGENET1K_V1)
        
        # Adjust conv1 if num_img_channels is not 3 (e.g. for grayscale)
        if num_img_channels != 3:
            original_conv1 = self.backbone.conv1
            self.backbone.conv1 = nn.Conv2d(num_img_channels, original_conv1.out_channels,
                                            kernel_size=original_conv1.kernel_size,
                                            stride=original_conv1.stride,
                                            padding=original_conv1.padding,
                                            bias=original_conv1.bias)
            # Basic weight initialization for the new conv1, could be more sophisticated
            if num_img_channels == 1: # Assuming original weights were for 3 channels (RGB)
                 # Average original weights across the R,G,B channels for the new single channel
                self.backbone.conv1.weight.data = original_conv1.weight.data.mean(dim=1, keepdim=True)


        in_feat = self.backbone.fc.in_features
        drop = 0.0 if overfit else 0.3
        self.backbone.fc = nn.Sequential(
            nn.Dropout(drop),
            nn.Linear(in_feat, 4),
            nn.Sigmoid()  # For normalized bbox coords [0,1]
        )

    def forward(self, x):
        return self.backbone(x)

class MotionStrengthModule(nn.Module):
    """
    Motion Strength Module inspired by MG-VTOD.
    Processes a sequence of frames to extract a motion map for the current frame.
    Input: Frame cuboid [B, T, C, H, W] (normalized RGB frames)
    Output: Motion map [B, 1, H, W] (normalized to [0,1])
    """
    def __init__(self, t_window=3, in_channels=3):
        super().__init__()
        # Temporal high-pass filter (Magno-like stage)
        # This 3D convolution processes the C input channels (e.g., RGB)
        # and outputs a single channel representing temporal differences.
        # Kernel size (t_window, 1, 1) means it only convolves along the time axis.
        self.temp_conv = nn.Conv3d(in_channels, 1, 
                                   kernel_size=(t_window, 1, 1),
                                   padding=(t_window // 2, 0, 0), 
                                   bias=False)
        
        # Spatial low-pass filter (Ganglion-like stage)
        # This 2D convolution smooths the temporally filtered map.
        self.spat_conv = nn.Conv2d(1, 1, 
                                   kernel_size=5, 
                                   padding=2, # (5-1)/2
                                   bias=False)
        # Optional: Initialize spat_conv weights to be an averaging filter
        # nn.init.constant_(self.spat_conv.weight, 1.0 / (5*5))

    def forward(self, frames_cuboid):
        # frames_cuboid: [B, T, C, H, W]
        # Permute to [B, C, T, H, W] for Conv3d compatibility
        x = frames_cuboid.permute(0, 2, 1, 3, 4) 

        # Apply temporal convolution
        m = self.temp_conv(x)  # Output: [B, 1, T, H, W]

        # Rectify (ReLU) and select the features for the latest time slice.
        # m[:, :, -1, :, :] corresponds to the processed output for the T-th (current) frame.
        m_current_time_slice = F.relu(m[:, :, -1, :, :])  # Output: [B, 1, H, W]
        
        # Apply spatial smoothing
        m_smooth = self.spat_conv(m_current_time_slice)  # Output: [B, 1, H, W]
        
        # Normalize to [0,1] to produce the motion strength map
        motion_map = torch.sigmoid(m_smooth) # Output: [B, 1, H, W]
        
        return motion_map

class TemporalResNet50Regressor(nn.Module):
    """
    Temporal UAV bounding box regressor.
    Uses a ResNet50 backbone and a MotionStrengthModule.
    Input: Frame cuboid [B, T, C, H, W]
    Output: Bounding box predictions [B, 4] for the current frame in the cuboid.
    """
    def __init__(self, seq_len=5, overfit=False, num_img_channels=3):
        super().__init__()
        self.seq_len = seq_len
        self.motion_module = MotionStrengthModule(t_window=seq_len, in_channels=num_img_channels)

        # Standard ResNet50 backbone
        self.backbone = models.resnet50(weights=models.ResNet50_Weights.IMAGENET1K_V1)

        # Modify the first convolutional layer of ResNet50 to accept C+1 channels:
        # num_img_channels (e.g., 3 for RGB) + 1 (for the motion map).
        original_conv1 = self.backbone.conv1
        self.backbone.conv1 = nn.Conv2d(num_img_channels + 1, 
                                        original_conv1.out_channels,
                                        kernel_size=original_conv1.kernel_size,
                                        stride=original_conv1.stride,
                                        padding=original_conv1.padding,
                                        bias=original_conv1.bias)
        
        # Copy weights from the original conv1 for the image channels.
        # Initialize weights for the new motion map channel (e.g., to zero or small random values).
        with torch.no_grad():
            self.backbone.conv1.weight[:, :num_img_channels, :, :] = original_conv1.weight.data
            self.backbone.conv1.weight[:, num_img_channels:, :, :].fill_(0.01) # Small init for motion channel

        # Replace the final fully connected layer for bounding box regression (4 coordinates)
        in_feat = self.backbone.fc.in_features
        drop = 0.0 if overfit else 0.3 # Dropout rate
        self.backbone.fc = nn.Sequential(
            nn.Dropout(drop),
            nn.Linear(in_feat, 4),
            nn.Sigmoid()  # For normalized bbox coordinates [0,1]
        )

    def forward(self, frame_cuboid):
        # frame_cuboid: [B, T, C, H, W], where T is seq_len
        
        # Extract the current frame (the last frame in the sequence)
        # Input shape for backbone: [B, C, H, W]
        current_frame = frame_cuboid[:, -1, :, :, :]  # Shape: [B, C, H, W]
        
        # Generate the motion map using the full cuboid.
        # The motion map is specific to the current frame but derived from the sequence.
        motion_map = self.motion_module(frame_cuboid)  # Shape: [B, 1, H, W]
        
        # Fuse the motion map with the current frame by concatenating along the channel dimension.
        fused_input = torch.cat([current_frame, motion_map], dim=1)  # Shape: [B, C+1, H, W]
        
        # Pass the fused input through the modified ResNet50 backbone.
        bbox_pred = self.backbone(fused_input)
        
        return bbox_pred
