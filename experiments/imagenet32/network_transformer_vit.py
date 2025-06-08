# File: network_transformer_vit.py

import torch
import torch.nn as nn
import torch.nn.functional as F

from torchcfm.models.unet.unet import UNetModelWrapper

##############################################################################
# Simple Patch Embedding (like in ViT)
##############################################################################
class PatchEmbed(nn.Module):
    """
    Splits the (B, C, H, W) feature map into non-overlapping patches and
    embeds each patch to `embed_dim`.
    """
    def __init__(
        self,
        in_channels=3,
        patch_size=4,
        embed_dim=128,
        image_size=(32, 32),
        include_pos_embed=True
    ):
        super().__init__()
        self.patch_size = patch_size
        self.embed_dim = embed_dim
        self.num_patches_h = image_size[0] // patch_size
        self.num_patches_w = image_size[1] // patch_size
        self.num_patches = self.num_patches_h * self.num_patches_w

        # Patch embedding via Conv2d
        self.patch_embed = nn.Conv2d(
            in_channels,
            embed_dim,
            kernel_size=patch_size,
            stride=patch_size
        )

        # Optional learnable positional embeddings
        self.include_pos_embed = include_pos_embed
        if include_pos_embed:
            self.pos_embed = nn.Parameter(
                torch.zeros(1, self.num_patches, embed_dim)
            )
        else:
            self.pos_embed = None

        # Initialize patch embedding weights
        nn.init.xavier_uniform_(self.patch_embed.weight)

    def forward(self, x):
        # x: (B, C, H, W)
        # => (B, E, H', W') after patch_embed
        x = self.patch_embed(x)  # shape: (B, embed_dim, H', W')
        B, E, Hp, Wp = x.shape
        # Flatten
        x = x.view(B, E, Hp * Wp).transpose(1, 2)  # => (B, N, E)
        # Add positional embedding if needed
        if self.pos_embed is not None:
            x = x + self.pos_embed  # (1, N, E) broadcast
        return x


##############################################################################
# Soft clamp
##############################################################################
def soft_clamp(x, clamp_val):
    """Tanh-based clamp: output in [-clamp_val, clamp_val]."""
    return clamp_val * torch.tanh(x / clamp_val)


##############################################################################
# Helper to create a dummy time tensor
##############################################################################
def dummy_time(x, value=0.5):
    """
    Create a (B,)-shaped tensor of `value`, matching x's device and dtype.
    """
    return torch.full(
        (x.shape[0],),
        value,
        device=x.device,
        dtype=x.dtype
    )


##############################################################################
# 1) EBM with a patch-based ViT head
##############################################################################
class EBViTModelWrapper(UNetModelWrapper):
    """
    Energy-Based Model with a patch-based ViT on top of the UNet output.
    Ignores the input time; always feeds a fixed dummy time to the UNet.
    
    Note: potential() and velocity() now accept (x, t) where t is ignored.
    """

    def __init__(
        self,
        dim=(3, 32, 32),
        num_channels=128,
        num_res_blocks=2,
        channel_mult=[1, 2, 2, 2],
        attention_resolutions="16",
        num_heads=4,
        num_head_channels=64,
        dropout=0.1,
        # UNet flags
        class_cond=False,
        learn_sigma=False,
        use_checkpoint=False,
        use_fp16=False,
        resblock_updown=False,
        use_scale_shift_norm=False,
        use_new_attention_order=False,
        # ViT-specific
        patch_size=4,
        embed_dim=128,
        transformer_nheads=4,
        transformer_nlayers=2,
        include_pos_embed=True,
        # EBM extras
        output_scale=1000.0,
        energy_clamp=None,
        **kwargs
    ):
        super().__init__(
            dim=dim,
            num_channels=num_channels,
            num_res_blocks=num_res_blocks,
            channel_mult=channel_mult,
            attention_resolutions=attention_resolutions,
            num_heads=num_heads,
            num_head_channels=num_head_channels,
            dropout=dropout,
            class_cond=class_cond,
            learn_sigma=learn_sigma,
            use_checkpoint=use_checkpoint,
            use_fp16=use_fp16,
            resblock_updown=resblock_updown,
            use_scale_shift_norm=use_scale_shift_norm,
            use_new_attention_order=use_new_attention_order,
            **kwargs
        )

        self.out_channels = dim[0]
        self.output_scale = output_scale
        self.energy_clamp = energy_clamp

        # 1) PatchEmbed for the UNet output
        self.patch_embed = PatchEmbed(
            in_channels=self.out_channels,
            patch_size=patch_size,
            embed_dim=embed_dim,
            image_size=dim[1:],  # (H, W)
            include_pos_embed=include_pos_embed
        )

        # 2) A small Transformer Encoder
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=embed_dim,
            nhead=transformer_nheads,
            dim_feedforward=4 * embed_dim,
            dropout=dropout,
            activation='gelu',
            batch_first=True
        )
        self.transformer_encoder = nn.TransformerEncoder(
            encoder_layer,
            num_layers=transformer_nlayers
        )

        # 3) Final linear => scalar
        self.final_linear = nn.Linear(embed_dim, 1)

    def potential(self, x, t):
        """
        Computes scalar potential V(x,t) => shape (B,).
        Ignores the provided time and always uses a fixed dummy time.
        """
        t_dummy = dummy_time(x, value=0.5)
        # UNet forward: shape (B, C, H, W)
        unet_out = super().forward(t_dummy, x)
        # Patch-embed: (B, N, embed_dim)
        tokens = self.patch_embed(unet_out)
        # Transformer: (B, N, embed_dim)
        encoded = self.transformer_encoder(tokens)
        # Mean-pool across tokens: (B, embed_dim)
        pooled = encoded.mean(dim=1)
        # Final linear to scalar: (B, 1) -> (B,)
        V = self.final_linear(pooled).view(-1)
        V = V * self.output_scale
        if self.energy_clamp is not None:
            V = soft_clamp(V, self.energy_clamp)
        return V

    def velocity(self, x, t):
        """
        Computes -∂V/∂x => shape (B, C, H, W).
        Ignores the provided time.
        """
        with torch.enable_grad():
            x = x.clone().detach().requires_grad_(True)
            V = self.potential(x, t)
            dVdx = torch.autograd.grad(
                outputs=V,
                inputs=x,
                grad_outputs=torch.ones_like(V),
                create_graph=True
            )[0]
            return -dVdx

    def forward(self, t, x, return_potential=False, *args, **kwargs):
        """
        Forward pass accepts a time tensor and an input tensor.
        The provided time is ignored (dummy time is used internally).
        If return_potential=True, returns V(x,t); otherwise returns velocity.
        """
        if return_potential:
            return self.potential(x, t)
        else:
            return self.velocity(x, t)


##############################################################################
# 2) EBM with a simple MLP head (SiLU in the hidden layer)
##############################################################################
class EBMLPModelWrapper(UNetModelWrapper):
    """
    Energy-Based Model that extends the UNet code but uses a simple MLP head.
    Ignores the provided time; always feeds a fixed dummy time to the UNet.
    
    MLP architecture:
        Global average pool -> Linear -> SiLU -> Linear -> scalar V(x,t).
    """

    def __init__(
        self,
        dim=(3, 32, 32),
        num_channels=128,
        num_res_blocks=2,
        channel_mult=[1, 2, 2, 2],
        attention_resolutions="16",
        num_heads=4,
        num_head_channels=64,
        dropout=0.1,
        # UNet flags
        class_cond=False,
        learn_sigma=False,
        use_checkpoint=False,
        use_fp16=False,
        resblock_updown=False,
        use_scale_shift_norm=False,
        use_new_attention_order=False,
        # EBM extras
        output_scale=1000.0,
        energy_clamp=None,
        **kwargs
    ):
        super().__init__(
            dim=dim,
            num_channels=num_channels,
            num_res_blocks=num_res_blocks,
            channel_mult=channel_mult,
            attention_resolutions=attention_resolutions,
            num_heads=num_heads,
            num_head_channels=num_head_channels,
            dropout=dropout,
            class_cond=class_cond,
            learn_sigma=learn_sigma,
            use_checkpoint=use_checkpoint,
            use_fp16=use_fp16,
            resblock_updown=resblock_updown,
            use_scale_shift_norm=use_scale_shift_norm,
            use_new_attention_order=use_new_attention_order,
            **kwargs
        )

        self.out_channels = dim[0]
        self.output_scale = output_scale
        self.energy_clamp = energy_clamp

        # Global average pooling
        self.pool = nn.AdaptiveAvgPool2d((1, 1))

        # Simple MLP: [Flatten -> Linear -> SiLU -> Linear -> scalar]
        self.mlp = nn.Sequential(
            nn.Flatten(),                     # => (B, C)
            nn.Linear(self.out_channels, self.out_channels),
            nn.SiLU(),
            nn.Linear(self.out_channels, 1)   # => scalar output
        )

    def potential(self, x, t):
        """
        Computes scalar potential V(x,t) => shape (B,).
        Ignores the provided time.
        """
        t_dummy = dummy_time(x, value=0.5)
        # UNet forward: (B, C, H, W)
        unet_out = super().forward(t_dummy, x)
        # Global average pool: (B, C, 1, 1)
        pooled = self.pool(unet_out)
        # MLP: (B, 1) -> (B,)
        V = self.mlp(pooled).view(-1)
        V = V * self.output_scale
        if self.energy_clamp is not None:
            V = soft_clamp(V, self.energy_clamp)
        return V

    def velocity(self, x, t):
        """
        Computes -∂V/∂x => shape (B, C, H, W).
        Ignores the provided time.
        """
        with torch.enable_grad():
            x = x.clone().detach().requires_grad_(True)
            V = self.potential(x, t)
            dVdx = torch.autograd.grad(
                outputs=V,
                inputs=x,
                grad_outputs=torch.ones_like(V),
                create_graph=True
            )[0]
            return -dVdx

    def forward(self, t, x, return_potential=False, *args, **kwargs):
        """
        Forward pass accepts a time tensor and an input tensor.
        The provided time is ignored (dummy time is used internally).
        If return_potential=True, returns V(x,t); otherwise returns velocity.
        """
        if return_potential:
            return self.potential(x, t)
        else:
            return self.velocity(x, t)
