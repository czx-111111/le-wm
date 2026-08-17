import torch
import torch.nn as nn

torch.hub._validate_not_a_forked_repo=lambda a,b,c: True

class VJepaV2Encoder(nn.Module):
    def __init__(self, name, feature_key):
        super().__init__()
        self.name = name

        # Load the vjepa2 model from torch.hub
        # This provides the get_vision_features() method needed for proper encoding
        self.image_embedder, _ = torch.hub.load('facebookresearch/vjepa2', 'vjepa2_vit_large')

        self.feature_key = feature_key

        # Get embedding dimension from the model
        # vjepa2-vitl typically has 1024 feature dimension
        self.emb_dim = self.image_embedder.embed_dim if hasattr(self.image_embedder, 'embed_dim') else 1024

        # Set latent dimensionality based on feature key
        if feature_key == "x_norm_patchtokens":
            self.latent_ndim = 2
        elif feature_key == "x_norm_clstoken":
            self.latent_ndim = 1
        else:
            raise ValueError(f"Invalid feature key: {feature_key}")

        # Get patch size from the model (vjepa2-vitl uses 16x16 patch size)
        self.patch_size = self.image_embedder.patch_embed.patch_size if hasattr(self.image_embedder, 'patch_embed') else 16

    # TODO: note we're currently undoing the default transform and reapplying the
    # image transform
    def forward(self, x):
        """
        Forward pass for encoding images.

        Args:
            x: Input tensor of shape (B, C, H, W) normalized with default_transform
               (mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5], range [-1, 1])

        Returns:
            emb: Encoded features of shape (B, num_patches, emb_dim) or (B, 1, emb_dim)
        """
        # Re-normalize from default transform to ImageNet normalization
        # Step 1: Convert from [-1, 1] back to [0, 1]
        # Current: output = (input - 0.5) / 0.5, so input = output * 0.5 + 0.5
        x = x * 0.5 + 0.5

        # Step 2: Apply ImageNet normalization
        # ImageNet: mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
        mean = torch.tensor([0.485, 0.456, 0.406], device=x.device).view(1, 3, 1, 1)
        std = torch.tensor([0.229, 0.224, 0.225], device=x.device).view(1, 3, 1, 1)
        x = (x - mean) / std

        # VJepa2 expects input with temporal dimension: (B, 1, C, H, W)
        # Add temporal dimension of 1 since we're processing single frames

        # copy along time dimension -- github issue on repo where an author suggested this

        # Get vision features using the vjepa2 model
        emb = self.image_embedder(x.unsqueeze(2).repeat(1,1,2,1,1))  # (B, num_patches, emb_dim)

        return emb

class VJepaV2EncoderAggregated(nn.Module):
    def __init__(self, name):
        super().__init__()
        self.name = name

        # Load the vjepa2 model from torch.hub
        self.image_embedder, _ = torch.hub.load('facebookresearch/vjepa2', 'vjepa2_vit_large')

        # Get embedding dimension from the model
        self.emb_dim = self.image_embedder.embed_dim if hasattr(self.image_embedder, 'embed_dim') else 1024
        self.feature_key = "x_norm_patchtokens"
        self.latent_ndim = 1

        # Get patch size from the model
        self.patch_size = self.image_embedder.patch_embed.patch_size if hasattr(self.image_embedder, 'patch_embed') else 16

        # Freeze the base model parameters
        for param in self.image_embedder.parameters():
            param.requires_grad = False

        # Trainable attention head
        self.attention_head = nn.Linear(self.emb_dim, 1)

    def forward(self, x):
        """
        Forward pass with aggregated features.

        Args:
            x: Input tensor of shape (B, C, H, W) normalized with default_transform

        Returns:
            aggregated: Aggregated features of shape (B, 1, emb_dim)
        """
        # Re-normalize from default transform to ImageNet normalization
        x = x * 0.5 + 0.5

        # Apply ImageNet normalization
        mean = torch.tensor([0.485, 0.456, 0.406], device=x.device).view(1, 3, 1, 1)
        std = torch.tensor([0.229, 0.224, 0.225], device=x.device).view(1, 3, 1, 1)
        x = (x - mean) / std

        # Extract features without computing gradients for efficiency
        with torch.no_grad():
            emb = self.image_embedder(x.unsqueeze(2).repeat(1,1,2,1,1))  # (B, num_patches, emb_dim)

        # Compute attention logits (trainable)
        attention_logits = self.attention_head(emb)  # (B, num_patches, 1)
        attention_logits = attention_logits.squeeze(-1)  # (B, num_patches)

        # Apply softmax to get attention weights
        attention_weights = torch.softmax(attention_logits, dim=1)  # (B, num_patches)

        # Apply weighted aggregation
        aggregated = torch.sum(emb * attention_weights.unsqueeze(-1), dim=1)  # (B, emb_dim)
        aggregated = aggregated.unsqueeze(1)  # (B, 1, emb_dim)

        return aggregated

class VJepaV2EncoderAggregatedWithBias(nn.Module):
    def __init__(self, name, img_size=224):
        super().__init__()
        self.name = name

        # Load the vjepa2 model from torch.hub
        self.image_embedder, _ = torch.hub.load('facebookresearch/vjepa2', 'vjepa2_vit_large')

        # Get embedding dimension from the model
        self.emb_dim = self.image_embedder.embed_dim if hasattr(self.image_embedder, 'embed_dim') else 1024
        self.feature_key = "x_norm_patchtokens"
        self.latent_ndim = 1

        # Get patch size from the model
        self.patch_size = self.image_embedder.patch_embed.patch_size if hasattr(self.image_embedder, 'patch_embed') else 16

        # Calculate number of patches based on image size
        self.img_size = img_size
        self.num_patches = (img_size // self.patch_size) ** 2

        # Freeze the base model parameters
        for param in self.image_embedder.parameters():
            param.requires_grad = False

        # Trainable attention head
        self.attention_head = nn.Linear(self.emb_dim, 1, bias=False)

        # Learnable positional bias/embedding
        # Shape: (1, num_patches, emb_dim) to be added to each batch
        self.positional_bias = nn.Parameter(torch.zeros(self.num_patches, self.emb_dim))
        # Initialize with small random values
        nn.init.normal_(self.positional_bias, mean=0.0, std=0.02)

    def forward(self, x):
        """
        Forward pass with aggregated features and learnable positional bias.

        Args:
            x: Input tensor of shape (B, C, H, W) normalized with default_transform

        Returns:
            aggregated: Aggregated features of shape (B, 1, emb_dim)
        """
        # Re-normalize from default transform to ImageNet normalization
        x = x * 0.5 + 0.5

        # Apply ImageNet normalization
        mean = torch.tensor([0.485, 0.456, 0.406], device=x.device).view(1, 3, 1, 1)
        std = torch.tensor([0.229, 0.224, 0.225], device=x.device).view(1, 3, 1, 1)
        x = (x - mean) / std

        # Extract features without computing gradients for efficiency
        with torch.no_grad():
            emb = self.image_embedder(x.unsqueeze(2).repeat(1,1,2,1,1))  # (B, num_patches, emb_dim)


        # Compute attention logits (trainable)
        attention_logits = self.attention_head(emb + self.positional_bias)  # (B, num_patches, 1)
        attention_logits = attention_logits.squeeze(-1)  # (B, num_patches)

        # Apply softmax to get attention weights
        attention_weights = torch.softmax(attention_logits, dim=1)  # (B, num_patches)

        # Apply weighted aggregation
        aggregated = torch.sum(emb * attention_weights.unsqueeze(-1), dim=1)  # (B, emb_dim)
        aggregated = aggregated.unsqueeze(1)  # (B, 1, emb_dim)

        return aggregated

class VJepaV2LowdimEncoder(nn.Module):
    def __init__(self, name, output_dim):
        super().__init__()
        self.name = name

        # Load the vjepa2 model from torch.hub
        self.image_embedder, _ = torch.hub.load('facebookresearch/vjepa2', 'vjepa2_vit_large')


        # Embedding dimension now output of final linear map
        self.emb_dim = output_dim
        self.latent_ndim = 2

        for param in self.image_embedder.parameters():
            param.requires_grad = False


        # Get patch size from the model
        self.patch_size = self.image_embedder.patch_embed.patch_size if hasattr(self.image_embedder, 'patch_embed') else 16

        # Linear layer to project to lower dimension
        self.projection = nn.Linear(self.image_embedder.embed_dim, output_dim)

    def forward(self, x):
        """
        Forward pass with low-dimensional projection.

        Args:
            x: Input tensor of shape (B, C, H, W) normalized with default_transform

        Returns:
            emb: Projected features of shape (B, num_patches, output_dim) or (B, 1, output_dim)
        """
        # Re-normalize from default transform to ImageNet normalization
        x = x * 0.5 + 0.5

        # Apply ImageNet normalization
        mean = torch.tensor([0.485, 0.456, 0.406], device=x.device).view(1, 3, 1, 1)
        std = torch.tensor([0.229, 0.224, 0.225], device=x.device).view(1, 3, 1, 1)
        x = (x - mean) / std

        # Get vision features using the vjepa2 model
        emb = self.image_embedder(x.unsqueeze(2).repeat(1,1,2,1,1))  # (B, num_patches, emb_dim)

        # Apply linear projection to the last dimension
        emb = self.projection(emb)  # (B, num_patches, output_dim)

        return emb
