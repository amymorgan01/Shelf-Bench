"""
DinoV3 model for ICE-BENCH: 

Model: ViT models pretrained on satellite dataset (SAT-493M) - all optical:
ViT-L/16 distilled	300M	SAT-493M
dinov3_vitl16
"""
import os
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Tuple


class DINOv3SegmentationModel(nn.Module):
    """
    DINOv3-based segmentation model with customizable segmentation head
    """
    
    def __init__(
        self, 
        num_classes: int, 
        img_size: int = 518,
        satellite_weights_path: Optional[str] = None,
        segmentation_head: str = "unet",  # "fcn", "unet", "fpn"
        freeze_backbone: bool = False
    ):
        super(DINOv3SegmentationModel, self).__init__()
        
        self.num_classes = num_classes
        self.img_size = img_size
        self.patch_size = 16  # DINOv3 ViT-L/16
        self.embed_dim = 1024  # ViT-L embedding dimension
        
        # Calculate feature map size after patch embedding
        self.feat_h = self.feat_w = img_size // self.patch_size
        
        # Load DINOv3 backbone
        self.backbone = self._load_dinov3_backbone(satellite_weights_path)
        
        if freeze_backbone:
            for param in self.backbone.parameters():
                param.requires_grad = False
        
        # Create segmentation head
        if segmentation_head == "fcn":
            self.seg_head = self._create_fcn_head()
        elif segmentation_head == "unet":
            self.seg_head = self._create_unet_head()
        elif segmentation_head == "fpn":
            self.seg_head = self._create_fpn_head()
        else:
            raise ValueError(f"Unknown segmentation head: {segmentation_head}")
    
    def _load_dinov3_backbone(self, satellite_weights_path: Optional[str] = None):
        """Load DINOv3 backbone with optional satellite weights"""
        DINOV3_GITHUB_LOCATION = "facebookresearch/dinov3"
        
        if os.getenv("DINOV3_LOCATION") is not None:
            DINOV3_LOCATION = os.getenv("DINOV3_LOCATION")
        else:
            DINOV3_LOCATION = DINOV3_GITHUB_LOCATION
        
        # Load model architecture
        backbone = torch.hub.load(
            repo_or_dir=DINOV3_LOCATION,
            model="dinov3_vitl16",
            pretrained=False if satellite_weights_path else True,
            source="local" if DINOV3_LOCATION != DINOV3_GITHUB_LOCATION else "github"
        )
        
        # Load satellite weights if provided
        if satellite_weights_path and os.path.exists(satellite_weights_path):
            print(f"Loading satellite pretrained weights from {satellite_weights_path}")
            satellite_weights = torch.load(satellite_weights_path, map_location='cpu')
            
            # Handle different weight formats
            if 'model' in satellite_weights:
                state_dict = satellite_weights['model']
            elif 'state_dict' in satellite_weights:
                state_dict = satellite_weights['state_dict']
            else:
                state_dict = satellite_weights
            
            # Clean state dict keys
            cleaned_state_dict = {}
            for key, value in state_dict.items():
                clean_key = key
                for prefix in ['module.', 'backbone.', 'encoder.', 'model.']:
                    if clean_key.startswith(prefix):
                        clean_key = clean_key[len(prefix):]
                        break
                cleaned_state_dict[clean_key] = value
            
            backbone.load_state_dict(cleaned_state_dict, strict=False)
            print("Successfully loaded satellite pretrained weights!")
        
        return backbone
    
    def _create_fcn_head(self):
        """Create a simple FCN-style segmentation head"""
        return nn.Sequential(
            nn.Conv2d(self.embed_dim, 512, kernel_size=3, padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True),
            nn.Dropout2d(0.1),
            nn.Conv2d(512, 256, kernel_size=3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.Dropout2d(0.1),
            nn.Conv2d(256, self.num_classes, kernel_size=1)
        )
    
    def _create_unet_head(self):
        """Create a U-Net style decoder"""
        return UNetDecoder(
            encoder_channels=[self.embed_dim],
            decoder_channels=[512, 256, 128, 64],
            n_blocks=4,
            num_classes=self.num_classes
        )
    
    @staticmethod
    def sar_to_rgb_channels(sar_image: torch.Tensor, method: str = "repeat") -> torch.Tensor:
        """
        Convert single-channel SAR to 3-channel format
        
        Args:
            sar_image: Single channel SAR image [B, 1, H, W] or [B, H, W]
            method: Conversion method - "repeat" [SAR, SAR, SAR]
        
        Returns:
            3-channel tensor [B, 3, H, W]
        """
        if sar_image.dim() == 3:
            sar_image = sar_image.unsqueeze(1)  # Add channel dimension
        
        if method == "repeat":
            # Simple repetition - most common approach
            return sar_image.repeat(1, 3, 1, 1)

    
    def _create_fpn_head(self):
        """Create an FPN-style segmentation head"""
        return FPNDecoder(
            encoder_channels=[self.embed_dim],
            pyramid_channels=256,
            segmentation_channels=128,
            num_classes=self.num_classes
        )
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, C, H, W = x.shape

        # Convert SAR to RGB
        if C == 1:
            x = self.sar_to_rgb_channels(x, method="repeat")
        
        # Extract features using DINOv3 backbone
        # Get patch embeddings (excluding CLS token)
        # with torch.no_grad() if hasattr(self, '_freeze_backbone') else torch.enable_grad():
        #     features = self.backbone.forward_features(x)
        #     # Remove CLS token and reshape to feature map
        #     patch_features = features[:, 1:]  # Remove CLS token
        #     patch_features = patch_features.reshape(B, self.feat_h, self.feat_w, self.embed_dim)
        #     patch_features = patch_features.permute(0, 3, 1, 2)  # B, C, H, W
        
        # Extract features using DINOv3 backbone
    # Get patch embeddings (excluding CLS token)
        # Extract features using DINOv3 backbone
        with torch.no_grad() if hasattr(self, '_freeze_backbone') else torch.enable_grad():
            features = self.backbone.forward_features(x)
            
            # Extract patch tokens (we know it's 'x_norm_patchtokens' from debug output)
            patch_features = features['x_norm_patchtokens']  # Shape: [B, N, D]
            
            print(f"Patch features shape: {patch_features.shape}")
            
            # Reshape to feature map
            # patch_features is [B, N, D] where N = feat_h * feat_w
            patch_features = patch_features.reshape(B, self.feat_h, self.feat_w, self.embed_dim)
            patch_features = patch_features.permute(0, 3, 1, 2)  # B, C, H, W
            
            print(f"Reshaped patch features: {patch_features.shape}")
        
        # Apply segmentation head - need to pass as list for UNet decoder
        seg_logits = self.seg_head([patch_features])  # Pass as list
        
        # Upsample to original image size
        seg_logits = F.interpolate(
            seg_logits, 
            size=(H, W), 
            mode='bilinear', 
            align_corners=False
        )
        
        return seg_logits


class UNetDecoder(nn.Module):
    """U-Net style decoder for DINOv3 features"""
    
    def __init__(self, encoder_channels, decoder_channels, n_blocks, num_classes):
        super().__init__()
        
        # Reverse to match U-Net convention
        encoder_channels = encoder_channels[::-1]
        
        # Computing blocks input and output channels
        head_channels = encoder_channels[0]
        in_channels = [head_channels] + list(decoder_channels[:-1])
        skip_channels = list(encoder_channels[1:]) + [0] * (n_blocks - len(encoder_channels) + 1)
        out_channels = decoder_channels
        
        self.center = nn.Identity()
        
        # Combine decoder keyword arguments
        blocks = [
            DecoderBlock(in_ch, skip_ch, out_ch)
            for in_ch, skip_ch, out_ch in zip(in_channels, skip_channels, out_channels)
        ]
        self.blocks = nn.ModuleList(blocks)
        
        self.final_conv = nn.Conv2d(decoder_channels[-1], num_classes, kernel_size=1)
    
    def forward(self, features):
        # features = features[-1]  # Take the deepest features
        # x = self.center(features)
        
        # for i, decoder_block in enumerate(self.blocks):
        #     skip = None  # No skip connections from encoder for single feature input
        #     x = decoder_block(x, skip)
            
        # x = self.final_conv(x)
    # Handle list input
        if isinstance(features, list):
            features = features[0]  # Take the first feature map
        
        x = self.center(features)
        
        for i, decoder_block in enumerate(self.blocks):
            skip = None  # No skip connections for single feature input
            x = decoder_block(x, skip)
            
        x = self.final_conv(x)


        return x


class DecoderBlock(nn.Module):
    def __init__(self, in_channels, skip_channels, out_channels):
        super().__init__()
        self.conv1 = nn.Conv2d(
            in_channels + skip_channels,
            out_channels,
            kernel_size=3,
            padding=1,
        )
        self.conv2 = nn.Conv2d(
            out_channels,
            out_channels,
            kernel_size=3,
            padding=1,
        )
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.bn2 = nn.BatchNorm2d(out_channels)
        
    def forward(self, x, skip=None):
        x = F.interpolate(x, scale_factor=2, mode="nearest")
        
        if skip is not None:
            x = torch.cat([x, skip], dim=1)
            
        x = F.relu(self.bn1(self.conv1(x)))
        x = F.relu(self.bn2(self.conv2(x)))
        
        return x


class FPNDecoder(nn.Module):
    """FPN-style decoder for DINOv3 features"""
    
    def __init__(self, encoder_channels, pyramid_channels, segmentation_channels, num_classes):
        super().__init__()
        
        self.lateral_conv = nn.Conv2d(encoder_channels[0], pyramid_channels, 1)
        
        # Segmentation head
        self.seg_blocks = nn.ModuleList([
            nn.Sequential(
                nn.Conv2d(pyramid_channels, segmentation_channels, 3, padding=1),
                nn.BatchNorm2d(segmentation_channels),
                nn.ReLU(inplace=True),
            )
        ])
        
        self.final_conv = nn.Conv2d(segmentation_channels, num_classes, 1)
        
    def forward(self, features):
        features = features[-1]  # Take the deepest features
        
        # Lateral connection
        fpn_feature = self.lateral_conv(features)
        
        # Segmentation
        seg_feature = self.seg_blocks[0](fpn_feature)
        output = self.final_conv(seg_feature)
        
        return output


def create_dinov3_segmentation_model(
    num_classes: int,
    img_size: int = 518,
    satellite_weights_path: Optional[str] = None,
    segmentation_head: str = "fcn",
    freeze_backbone: bool = False,
    in_channels: int = 3
) -> DINOv3SegmentationModel:
    """
    Factory function to create DINOv3 segmentation model
    
    Args:
        num_classes: Number of segmentation classes
        img_size: Input image size
        satellite_weights_path: Path to satellite pretrained weights
        segmentation_head: Type of segmentation head ("fcn", "unet", "fpn")
        freeze_backbone: Whether to freeze backbone parameters
        in_channels: Number of input channels (for future compatibility)
    
    Returns:
        DINOv3SegmentationModel instance
    """
    
    if in_channels != 3:
        print(f"Warning: DINOv3 expects 3 input channels, got {in_channels}. "
              "Consider preprocessing your data or using a different model.")
    
    model = DINOv3SegmentationModel(
        num_classes=num_classes,
        img_size=img_size,
        satellite_weights_path=satellite_weights_path,
        segmentation_head=segmentation_head,
        freeze_backbone=freeze_backbone
    )
    
    return model


# # initial set 

# DINOV3_GITHUB_LOCATION = "facebookresearch/dinov3"

# if os.getenv("DINOV3_LOCATION") is not None:
#     DINOV3_LOCATION = os.getenv("DINOV3_LOCATION")
# else:
#     DINOV3_LOCATION = DINOV3_GITHUB_LOCATION

# print(f"DINOv3 location set to {DINOV3_LOCATION}")

# MODEL_DINOV3_VITL = "dinov3_vitl16"
# MODEL_NAME = MODEL_DINOV3_VITL

# # Path to your satellite-pretrained weights
# SATELLITE_WEIGHTS_PATH = "/home/users/amorgan/benchmark_CB_AM/models/dinov3_vitl16_pretrain_sat493m-eadcf0ff.pth"

# def load_dinov3_with_satellite_weights():
#     """Load DINOv3 model with satellite pretrained weights"""
    
#     # Load model architecture without pretrained weights
#     print("Loading DINOv3 model architecture...")
#     model = torch.hub.load(
#         repo_or_dir=DINOV3_LOCATION,
#         model=MODEL_NAME,
#         pretrained=False,  # Don't load standard pretrained weights
#         source="local" if DINOV3_LOCATION != DINOV3_GITHUB_LOCATION else "github"
#     )
    
#     # Load the satellite pretrained weights
#     print(f"Loading satellite pretrained weights from {SATELLITE_WEIGHTS_PATH}...")
    
#     if not os.path.exists(SATELLITE_WEIGHTS_PATH):
#         raise FileNotFoundError(f"Satellite weights file not found: {SATELLITE_WEIGHTS_PATH}")
    
#     # Load the state dict
#     satellite_weights = torch.load(SATELLITE_WEIGHTS_PATH, map_location='cpu')
    
#     # Handle different possible formats of saved weights
#     if 'model' in satellite_weights:
#         state_dict = satellite_weights['model']
#     elif 'state_dict' in satellite_weights:
#         state_dict = satellite_weights['state_dict']
#     else:
#         state_dict = satellite_weights
    
#     # Remove any prefix if present (e.g., 'module.', 'backbone.', etc.)
#     cleaned_state_dict = {}
#     for key, value in state_dict.items():
#         # Remove common prefixes
#         clean_key = key
#         for prefix in ['module.', 'backbone.', 'encoder.', 'model.']:
#             if clean_key.startswith(prefix):
#                 clean_key = clean_key[len(prefix):]
#                 break
#         cleaned_state_dict[clean_key] = value
    
#     # Load the weights into the model
#     missing_keys, unexpected_keys = model.load_state_dict(cleaned_state_dict, strict=False)
    
#     if missing_keys:
#         print(f"Warning: Missing keys in state dict: {missing_keys}")
#     if unexpected_keys:
#         print(f"Warning: Unexpected keys in state dict: {unexpected_keys}")
    
#     print("Successfully loaded satellite pretrained weights!")
#     return model

# # Load the model
# try:
#     model = load_dinov3_with_satellite_weights()
#     model.eval()  # Set to evaluation mode
#     print("Model loaded successfully and set to evaluation mode")
    
#     # Optional: Move to GPU if available
#     device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
#     model = model.to(device)
#     print(f"Model moved to device: {device}")
    
# except Exception as e:
#     print(f"Error loading model: {str(e)}")
#     print("Trying alternative loading method...")
    
#     # Alternative: Load with pretrained=True first, then replace weights
#     try:
#         model = torch.hub.load(
#             repo_or_dir=DINOV3_LOCATION,
#             model=MODEL_NAME,
#             pretrained=True,  # Load with standard weights first
#             source="local" if DINOV3_LOCATION != DINOV3_GITHUB_LOCATION else "github"
#         )
        
#         # Then load satellite weights
#         satellite_weights = torch.load(SATELLITE_WEIGHTS_PATH, map_location='cpu')
        
#         if 'model' in satellite_weights:
#             state_dict = satellite_weights['model']
#         elif 'state_dict' in satellite_weights:
#             state_dict = satellite_weights['state_dict']
#         else:
#             state_dict = satellite_weights
            
#         # Clean state dict keys
#         cleaned_state_dict = {}
#         for key, value in state_dict.items():
#             clean_key = key
#             for prefix in ['module.', 'backbone.', 'encoder.', 'model.']:
#                 if clean_key.startswith(prefix):
#                     clean_key = clean_key[len(prefix):]
#                     break
#             cleaned_state_dict[clean_key] = value
        
#         model.load_state_dict(cleaned_state_dict, strict=False)
#         model.eval()
        
#         device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
#         model = model.to(device)
        
#         print("Model loaded successfully using alternative method!")
        
#     except Exception as e2:
#         print(f"Alternative loading method also failed: {str(e2)}")
#         raise e2


# print("\nModel setup complete! You can now use the model for feature extraction.")
