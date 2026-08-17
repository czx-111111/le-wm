import torch
import torch.nn as nn

torch.hub._validate_not_a_forked_repo=lambda a,b,c: True

class DinoV2Encoder(nn.Module):
    def __init__(self, name, feature_key):
        super().__init__()
        self.name = name
        self.base_model = torch.hub.load("facebookresearch/dinov2", name)
        self.feature_key = feature_key
        self.emb_dim = self.base_model.num_features
        if feature_key == "x_norm_patchtokens":
            self.latent_ndim = 2
        elif feature_key == "x_norm_clstoken":
            self.latent_ndim = 1
        else:
            raise ValueError(f"Invalid feature key: {feature_key}")

        self.patch_size = self.base_model.patch_size

    def forward(self, x):
        emb = self.base_model.forward_features(x)[self.feature_key]
        if self.latent_ndim == 1:
            emb = emb.unsqueeze(1) # dummy patch dim
        return emb

class DinoV2EncoderAggregated(nn.Module):
    def __init__(self, name):
        super().__init__()
        self.name = name
        self.base_model = torch.hub.load("facebookresearch/dinov2", name)
        self.emb_dim = self.base_model.num_features
        self.feature_key = "x_norm_patchtokens"
        self.latent_ndim = 1

        self.patch_size = self.base_model.patch_size

        # Freeze the base model parameters
        for param in self.base_model.parameters():
            param.requires_grad = False

        # Trainable attention head
        self.attention_head = nn.Linear(self.emb_dim, 1)

    def forward(self, x):
        # Extract features without computing gradients for efficiency
        with torch.no_grad():
            emb = self.base_model.forward_features(x)[self.feature_key]
        # emb shape: (B, 196, 384)

        # Compute attention logits (trainable)
        attention_logits = self.attention_head(emb)  # (B, 196, 1)
        attention_logits = attention_logits.squeeze(-1)  # (B, 196)

        # Apply softmax to get attention weights
        attention_weights = torch.softmax(attention_logits, dim=1)  # (B, 196)

        # Apply weighted aggregation
        aggregated = torch.sum(emb * attention_weights.unsqueeze(-1), dim=1)  # (B, 384)
        aggregated = aggregated.unsqueeze(1)  # (B, 1, 384)

        return aggregated
