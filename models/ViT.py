"""
Vision transformer for ICE-BENCH
Using the model: /home/users/amorgan/benchmark_CB_AM/models/ViT-L_16.npz from imagenet21k pre-train + imagenet2012 fine-tuned models (roughly 1 GB)
downloaded via: wget https://storage.googleapis.com/vit_models/imagenet21k+imagenet2012/ViT-L_16.npz

TODO: This already makes patches, so it will take the 256x256 image, and divide it into 16x16 patches.


"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from torch.nn import Dropout, Softmax, Linear, Conv2d, LayerNorm
import copy
import math
from torchvision.models import vit_l_16, ViT_L_16_Weights


# Configuration class for ViT-L/16
class VisionTransformerConfig:
    def __init__(self):
        # Model architecture
        self.hidden_size = 1024  # ViT-Large
        self.transformer = {
            "mlp_dim": 4096,
            "num_heads": 16,
            "num_layers": 24,
            "attention_dropout_rate": 0.0,
            "dropout_rate": 0.1,
        }
        self.classifier = "token"
        self.representation_size = None
        
        # Patch settings for ViT-L/16
        self.patches = {
            "size": (16, 16),
        }
        
        # Training settings (can be adjusted)
        self.n_classes = 1000  # ImageNet classes,
        self.resnet = None

# In your ViT.py file, update the ViTSegmentation class:

class ViTSegmentation(nn.Module):
    def __init__(self, num_classes=2, img_size=256, use_pretrained=True, in_channels=1):
        super(ViTSegmentation, self).__init__()
        
        # Create the ViT backbone
        if use_pretrained:
            weights = ViT_L_16_Weights.IMAGENET1K_V1
            # Use standard 224 size for pretrained weights
            self.vit = vit_l_16(weights=weights)
            self.vit_input_size = 224
        else:
            self.vit = vit_l_16(weights=None, image_size=img_size)
            self.vit_input_size = img_size
        
        # Store original settings
        self.img_size = img_size
        self.patch_size = 16
        self.num_patches_per_side = self.vit_input_size // self.patch_size
        
        # Get feature dimension and replace head
        feature_dim = self.vit.heads.head.in_features  # Should be 1024 for ViT-L
        self.vit.heads = nn.Identity()
        
        # Handle single channel input by replacing the first conv layer
        if in_channels == 1:
            original_conv = self.vit.conv_proj
            self.vit.conv_proj = nn.Conv2d(
                in_channels=1,
                out_channels=original_conv.out_channels,
                kernel_size=original_conv.kernel_size,
                stride=original_conv.stride,
                padding=original_conv.padding,
                bias=original_conv.bias is not None
            )
            
            # Initialize the new conv layer properly
            if use_pretrained:
                with torch.no_grad():
                    # Average the RGB weights for grayscale
                    self.vit.conv_proj.weight.copy_(
                        original_conv.weight.mean(dim=1, keepdim=True)
                    )
                    if original_conv.bias is not None:
                        self.vit.conv_proj.bias.copy_(original_conv.bias)
        
        # Segmentation decoder
        self.decoder = nn.Sequential(
            nn.Linear(feature_dim, 512),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(256, num_classes)
        )
        
    def extract_patch_features(self, x):
        """Extract patch features from ViT backbone without using the full forward pass"""
        B = x.shape[0]
        
        # Manual feature extraction to avoid issues with modified ViT
        # 1. Patch embedding
        x = self.vit.conv_proj(x)  # [B, hidden_dim, H_patches, W_patches]
        x = x.flatten(2).transpose(1, 2)  # [B, num_patches, hidden_dim]
        
        # 2. Add class token and position embeddings
        cls_token = self.vit.class_token.expand(B, -1, -1)
        x = torch.cat([cls_token, x], dim=1)
        x = x + self.vit.encoder.pos_embedding
        x = self.vit.encoder.dropout(x)
        
        # 3. Pass through transformer blocks
        for layer in self.vit.encoder.layers:
            x = layer(x)
        
        # 4. Apply final layer norm
        x = self.vit.encoder.ln(x)
        
        return x
        
    def forward(self, x):
        B, C, H, W = x.shape
        
        # Resize input if needed for pretrained model
        if self.vit_input_size != H:
            x_vit = F.interpolate(x, size=(self.vit_input_size, self.vit_input_size), 
                                mode='bilinear', align_corners=False)
        else:
            x_vit = x
        
        # Extract features using manual extraction
        features = self.extract_patch_features(x_vit)  # [B, num_patches + 1, feature_dim]
        
        # Remove CLS token to get only patch features
        patch_features = features[:, 1:]  # [B, num_patches, feature_dim]
        
        # Verify shapes
        expected_patches = self.num_patches_per_side ** 2
        if patch_features.shape[1] != expected_patches:
            raise RuntimeError(f"Expected {expected_patches} patches but got {patch_features.shape[1]}")
        
        # Apply decoder to each patch
        patch_logits = self.decoder(patch_features)  # [B, num_patches, num_classes]
        
        # Reshape to spatial grid
        patch_logits = patch_logits.reshape(
            B, self.num_patches_per_side, self.num_patches_per_side, -1
        ).permute(0, 3, 1, 2)  # [B, num_classes, H_patches, W_patches]
        
        # Upsample to target resolution
        output = F.interpolate(
            patch_logits, 
            size=(H, W), 
            mode='bilinear', 
            align_corners=False
        )
        
        return output


def swish(x):
    return x * torch.sigmoid(x)

ACT2FN = {"gelu": torch.nn.functional.gelu, "relu": torch.nn.functional.relu, "swish": swish}

class Attention(nn.Module):
    def __init__(self, config):
        super(Attention, self).__init__()
        self.num_attention_heads = config.transformer["num_heads"]
        self.attention_head_size = int(config.hidden_size / self.num_attention_heads)
        self.all_head_size = self.num_attention_heads * self.attention_head_size

        self.query = Linear(config.hidden_size, self.all_head_size)
        self.key = Linear(config.hidden_size, self.all_head_size)
        self.value = Linear(config.hidden_size, self.all_head_size)

        self.out = Linear(config.hidden_size, config.hidden_size)
        self.attn_dropout = Dropout(config.transformer["attention_dropout_rate"])
        self.proj_dropout = Dropout(config.transformer["attention_dropout_rate"])

        self.softmax = Softmax(dim=-1)

    def transpose_for_scores(self, x):
        new_x_shape = x.size()[:-1] + (self.num_attention_heads, self.attention_head_size)
        x = x.view(*new_x_shape)
        return x.permute(0, 2, 1, 3)

    def forward(self, hidden_states):
        mixed_query_layer = self.query(hidden_states)
        mixed_key_layer = self.key(hidden_states)
        mixed_value_layer = self.value(hidden_states)

        query_layer = self.transpose_for_scores(mixed_query_layer)
        key_layer = self.transpose_for_scores(mixed_key_layer)
        value_layer = self.transpose_for_scores(mixed_value_layer)

        attention_scores = torch.matmul(query_layer, key_layer.transpose(-1, -2))
        attention_scores = attention_scores / math.sqrt(self.attention_head_size)
        attention_probs = self.softmax(attention_scores)
        weights = attention_probs
        attention_probs = self.attn_dropout(attention_probs)

        context_layer = torch.matmul(attention_probs, value_layer)
        context_layer = context_layer.permute(0, 2, 1, 3).contiguous()
        new_context_layer_shape = context_layer.size()[:-2] + (self.all_head_size,)
        context_layer = context_layer.view(*new_context_layer_shape)
        attention_output = self.out(context_layer)
        attention_output = self.proj_dropout(attention_output)
        return attention_output, weights

class Mlp(nn.Module):
    def __init__(self, config):
        super(Mlp, self).__init__()
        self.fc1 = Linear(config.hidden_size, config.transformer["mlp_dim"])
        self.fc2 = Linear(config.transformer["mlp_dim"], config.hidden_size)
        self.act_fn = ACT2FN["gelu"]
        self.dropout = Dropout(config.transformer["dropout_rate"])

    def forward(self, x):
        x = self.fc1(x)
        x = self.act_fn(x)
        x = self.dropout(x)
        x = self.fc2(x)
        x = self.dropout(x)
        return x

class Block(nn.Module):
    def __init__(self, config):
        super(Block, self).__init__()
        self.hidden_size = config.hidden_size
        self.attention_norm = LayerNorm(config.hidden_size, eps=1e-6)
        self.ffn_norm = LayerNorm(config.hidden_size, eps=1e-6)
        self.ffn = Mlp(config)
        self.attn = Attention(config)

    def forward(self, x):
        h = x
        x = self.attention_norm(x)
        x, weights = self.attn(x)
        x = x + h

        h = x
        x = self.ffn_norm(x)
        x = self.ffn(x)
        x = x + h
        return x, weights

class Encoder(nn.Module):
    def __init__(self, config):
        super(Encoder, self).__init__()
        self.layer = nn.ModuleList()
        for _ in range(config.transformer["num_layers"]):
            layer = Block(config)
            self.layer.append(copy.deepcopy(layer))

    def forward(self, hidden_states):
        attn_weights = []
        for layer_block in self.layer:
            hidden_states, weights = layer_block(hidden_states)
            attn_weights.append(weights)
        encoded = LayerNorm(hidden_states.size(-1), eps=1e-6)(hidden_states)
        return encoded, attn_weights

class Embeddings(nn.Module):
    """Construct the embeddings from patch, position embeddings."""
    def __init__(self, config, img_size, in_channels=3):
        super(Embeddings, self).__init__()
        img_size = (img_size, img_size) if isinstance(img_size, int) else img_size
        patch_size = config.patches["size"]
        n_patches = (img_size[0] // patch_size[0]) * (img_size[1] // patch_size[1])

        self.patch_embeddings = Conv2d(in_channels=in_channels,
                                     out_channels=config.hidden_size,
                                     kernel_size=patch_size,
                                     stride=patch_size)
        self.position_embeddings = nn.Parameter(torch.zeros(1, n_patches+1, config.hidden_size))
        self.cls_token = nn.Parameter(torch.zeros(1, 1, config.hidden_size))

        self.dropout = Dropout(config.transformer["dropout_rate"])

    def forward(self, x):
        B = x.shape[0]
        cls_tokens = self.cls_token.expand(B, -1, -1)

        x = self.patch_embeddings(x)
        x = x.flatten(2)
        x = x.transpose(-1, -2)
        x = torch.cat((cls_tokens, x), dim=1)

        embeddings = x + self.position_embeddings
        embeddings = self.dropout(embeddings)
        return embeddings

class VisionTransformer(nn.Module):
    def __init__(self, config, img_size=256, num_classes=2, zero_head=False, in_channels=3):
        super(VisionTransformer, self).__init__()
        self.num_classes = num_classes
        self.zero_head = zero_head
        self.classifier = config.classifier

        self.embeddings = Embeddings(config, img_size=img_size, in_channels=in_channels)
        self.encoder = Encoder(config)

        self.head = Linear(config.hidden_size, num_classes)

    def forward(self, x, labels=None):
        x = self.embeddings(x)
        x, attn_weights = self.encoder(x)

        if self.classifier == "token":
            x = x[:, 0]
        elif self.classifier == "gap":
            x = torch.mean(x, dim=1)

        logits = self.head(x)

        if labels is not None:
            loss_fct = torch.nn.CrossEntropyLoss()
            loss = loss_fct(logits.view(-1, self.num_classes), labels.view(-1))
            return loss, logits, attn_weights
        else:
            return logits, attn_weights

    def load_from_npz(self, npz_path):
        """Load pre-trained weights from npz file"""
        weights = np.load(npz_path)
        
        with torch.no_grad():
            # Load embedding weights
            if 'embedding/kernel' in weights:
                kernel = torch.from_numpy(weights['embedding/kernel']).permute(3, 2, 0, 1)
                self.embeddings.patch_embeddings.weight.copy_(kernel)
            if 'embedding/bias' in weights:
                bias = torch.from_numpy(weights['embedding/bias'])
                self.embeddings.patch_embeddings.bias.copy_(bias)
            
            # Load position embeddings
            if 'Transformer/posembed_input/pos_embedding' in weights:
                pos_embed = torch.from_numpy(weights['Transformer/posembed_input/pos_embedding'])
                self.embeddings.position_embeddings.copy_(pos_embed)
            
            # Load class token
            if 'cls' in weights:
                cls_token = torch.from_numpy(weights['cls'])
                self.embeddings.cls_token.copy_(cls_token)
            
            # Load transformer layers
            for i, layer in enumerate(self.encoder.layer):
                # Attention weights
                prefix = f'Transformer/encoderblock_{i}/'
                
                # Query, Key, Value weights
                if f'{prefix}MultiHeadDotProductAttention_1/query/kernel' in weights:
                    q_kernel = torch.from_numpy(weights[f'{prefix}MultiHeadDotProductAttention_1/query/kernel']).transpose(0, 1)
                    layer.attn.query.weight.copy_(q_kernel)
                if f'{prefix}MultiHeadDotProductAttention_1/query/bias' in weights:
                    q_bias = torch.from_numpy(weights[f'{prefix}MultiHeadDotProductAttention_1/query/bias'])
                    layer.attn.query.bias.copy_(q_bias)
                
                if f'{prefix}MultiHeadDotProductAttention_1/key/kernel' in weights:
                    k_kernel = torch.from_numpy(weights[f'{prefix}MultiHeadDotProductAttention_1/key/kernel']).transpose(0, 1)
                    layer.attn.key.weight.copy_(k_kernel)
                if f'{prefix}MultiHeadDotProductAttention_1/key/bias' in weights:
                    k_bias = torch.from_numpy(weights[f'{prefix}MultiHeadDotProductAttention_1/key/bias'])
                    layer.attn.key.bias.copy_(k_bias)
                
                if f'{prefix}MultiHeadDotProductAttention_1/value/kernel' in weights:
                    v_kernel = torch.from_numpy(weights[f'{prefix}MultiHeadDotProductAttention_1/value/kernel']).transpose(0, 1)
                    layer.attn.value.weight.copy_(v_kernel)
                if f'{prefix}MultiHeadDotProductAttention_1/value/bias' in weights:
                    v_bias = torch.from_numpy(weights[f'{prefix}MultiHeadDotProductAttention_1/value/bias'])
                    layer.attn.value.bias.copy_(v_bias)
                
                # Output projection
                if f'{prefix}MultiHeadDotProductAttention_1/out/kernel' in weights:
                    out_kernel = torch.from_numpy(weights[f'{prefix}MultiHeadDotProductAttention_1/out/kernel']).transpose(0, 1)
                    layer.attn.out.weight.copy_(out_kernel)
                if f'{prefix}MultiHeadDotProductAttention_1/out/bias' in weights:
                    out_bias = torch.from_numpy(weights[f'{prefix}MultiHeadDotProductAttention_1/out/bias'])
                    layer.attn.out.bias.copy_(out_bias)
                
                # Layer norms
                if f'{prefix}LayerNorm_0/scale' in weights:
                    ln1_weight = torch.from_numpy(weights[f'{prefix}LayerNorm_0/scale'])
                    layer.attention_norm.weight.copy_(ln1_weight)
                if f'{prefix}LayerNorm_0/bias' in weights:
                    ln1_bias = torch.from_numpy(weights[f'{prefix}LayerNorm_0/bias'])
                    layer.attention_norm.bias.copy_(ln1_bias)
                
                if f'{prefix}LayerNorm_2/scale' in weights:
                    ln2_weight = torch.from_numpy(weights[f'{prefix}LayerNorm_2/scale'])
                    layer.ffn_norm.weight.copy_(ln2_weight)
                if f'{prefix}LayerNorm_2/bias' in weights:
                    ln2_bias = torch.from_numpy(weights[f'{prefix}LayerNorm_2/bias'])
                    layer.ffn_norm.bias.copy_(ln2_bias)
                
                # MLP weights
                if f'{prefix}MlpBlock_3/Dense_0/kernel' in weights:
                    mlp1_kernel = torch.from_numpy(weights[f'{prefix}MlpBlock_3/Dense_0/kernel']).transpose(0, 1)
                    layer.ffn.fc1.weight.copy_(mlp1_kernel)
                if f'{prefix}MlpBlock_3/Dense_0/bias' in weights:
                    mlp1_bias = torch.from_numpy(weights[f'{prefix}MlpBlock_3/Dense_0/bias'])
                    layer.ffn.fc1.bias.copy_(mlp1_bias)
                
                if f'{prefix}MlpBlock_3/Dense_1/kernel' in weights:
                    mlp2_kernel = torch.from_numpy(weights[f'{prefix}MlpBlock_3/Dense_1/kernel']).transpose(0, 1)
                    layer.ffn.fc2.weight.copy_(mlp2_kernel)
                if f'{prefix}MlpBlock_3/Dense_1/bias' in weights:
                    mlp2_bias = torch.from_numpy(weights[f'{prefix}MlpBlock_3/Dense_1/bias'])
                    layer.ffn.fc2.bias.copy_(mlp2_bias)
            
            # Load final layer norm (encoder norm)
            if 'Transformer/encoder_norm/scale' in weights:
                encoder_norm_weight = torch.from_numpy(weights['Transformer/encoder_norm/scale'])
                self.encoder.layer[-1] = LayerNorm(encoder_norm_weight.size(0), eps=1e-6)
                # Note: The encoder norm is applied in the Encoder class
            
            # Load classification head
            if 'head/kernel' in weights:
                head_kernel = torch.from_numpy(weights['head/kernel']).transpose(0, 1)
                if head_kernel.shape[0] != self.head.weight.shape[0]:
                    print(f"Warning: Head dimension mismatch. Expected {self.head.weight.shape[0]}, got {head_kernel.shape[0]}")
                    if self.zero_head or head_kernel.shape[0] == 21843:  # ImageNet-21k
                        print("Initializing head to zero or using different number of classes")
                        nn.init.zeros_(self.head.weight)
                        nn.init.zeros_(self.head.bias)
                    else:
                        self.head.weight.copy_(head_kernel)
                else:
                    self.head.weight.copy_(head_kernel)
            
            if 'head/bias' in weights and not self.zero_head:
                head_bias = torch.from_numpy(weights['head/bias'])
                if head_bias.shape[0] == self.head.bias.shape[0]:
                    self.head.bias.copy_(head_bias)

# def create_vit_large_16(num_classes=2, img_size=256, pretrained_path=None):
#     """Create ViT-Large/16 model"""
#     config = VisionTransformerConfig()
#     config.n_classes = num_classes
    
#     model = VisionTransformer(config, img_size=img_size, num_classes=num_classes)
    
#     if pretrained_path:
#         print(f"Loading pretrained weights from {pretrained_path}")
#         model.load_from_npz(pretrained_path)
    
#     return model

# Update your create function
def create_vit_large_16(num_classes=2, img_size=256, use_pretrained=True, in_channels=1):
    """Create ViT-Large/16 for segmentation"""
    return ViTSegmentation(
        num_classes=num_classes,
        img_size=img_size,
        use_pretrained=use_pretrained,
        in_channels=in_channels
    )


# # Example usage
# if __name__ == "__main__":
#     # Create model
#     model = create_vit_large_16(
#         num_classes=2,  # Adjust for your task
#         img_size=256,
#         pretrained_path="/home/users/amorgan/benchmark_CB_AM/models/ViT-L_16.npz"
#     )
    
    # # Test forward pass
    # dummy_input = torch.randn(1, 3, 224, 224)
    # with torch.no_grad():
    #     logits, attention_weights = model(dummy_input)
    
    # print(f"Model output shape: {logits.shape}")
    # print(f"Number of attention layers: {len(attention_weights)}")
    # print(f"Attention weight shape per layer: {attention_weights[0].shape}")