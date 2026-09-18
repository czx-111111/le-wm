import torch
from torch import nn
import torch.nn.functional as F
from einops import rearrange

def modulate(x, shift, scale):
    """AdaLN-zero modulation"""
    return x * (1 + scale) + shift

class SIGReg(torch.nn.Module):
    """Sketch Isotropic Gaussian Regularizer (single-GPU!)"""

    def __init__(self, knots=17, num_proj=1024):
        super().__init__()
        self.num_proj = num_proj
        t = torch.linspace(0, 3, knots, dtype=torch.float32)
        dt = 3 / (knots - 1)
        weights = torch.full((knots,), 2 * dt, dtype=torch.float32)
        weights[[0, -1]] = dt
        window = torch.exp(-t.square() / 2.0)
        self.register_buffer("t", t)
        self.register_buffer("phi", window)
        self.register_buffer("weights", weights * window)

    def forward(self, proj):
        """
        proj: (T, B, D)
        """
        # sample random projections
        A = torch.randn(proj.size(-1), self.num_proj, device=proj.device)
        A = A.div_(A.norm(p=2, dim=0))
        # compute the epps-pulley statistic
        x_t = (proj @ A).unsqueeze(-1) * self.t
        err = (x_t.cos().mean(-3) - self.phi).square() + x_t.sin().mean(-3).square()
        statistic = (err @ self.weights) * proj.size(-2)
        return statistic.mean() # average over projections and time
    
class FeedForward(nn.Module):
    """FeedForward network used in Transformers"""

    def __init__(self, dim, hidden_dim, dropout=0.0):
        super().__init__()
        self.net = nn.Sequential(
            nn.LayerNorm(dim),
            nn.Linear(dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, dim),
            nn.Dropout(dropout),
        )

    def forward(self, x):
        return self.net(x)


class Attention(nn.Module):
    """Scaled dot-product attention with causal masking"""

    def __init__(self, dim, heads=8, dim_head=64, dropout=0.0):
        super().__init__()
        inner_dim = dim_head * heads
        project_out = not (heads == 1 and dim_head == dim)
        self.heads = heads
        self.scale = dim_head**-0.5
        self.dropout = dropout
        self.norm = nn.LayerNorm(dim)
        self.attend = nn.Softmax(dim=-1)
        self.to_qkv = nn.Linear(dim, inner_dim * 3, bias=False)
        self.to_out = (
            nn.Sequential(nn.Linear(inner_dim, dim), nn.Dropout(dropout))
            if project_out
            else nn.Identity()
        )

    def forward(self, x, causal=True, attn_mask=None):
        """
        x : (B, T, D)
        attn_mask: optional additive mask broadcastable to (B, heads, T, T)
        """
        x = self.norm(x)
        drop = self.dropout if self.training else 0.0
        qkv = self.to_qkv(x).chunk(3, dim=-1)  # q, k, v: (B, heads, T, dim_head)
        q, k, v = (rearrange(t, "b t (h d) -> b h t d", h=self.heads) for t in qkv)
        use_causal = causal and attn_mask is None
        out = F.scaled_dot_product_attention(
            q, k, v, attn_mask=attn_mask, dropout_p=drop, is_causal=use_causal
        )
        out = rearrange(out, "b h t d -> b t (h d)")
        return self.to_out(out)


class ConditionalBlock(nn.Module):
    """Transformer block with AdaLN-zero conditioning"""

    def __init__(self, dim, heads, dim_head, mlp_dim, dropout=0.0):
        super().__init__()

        self.attn = Attention(dim, heads=heads, dim_head=dim_head, dropout=dropout)
        self.mlp = FeedForward(dim, mlp_dim, dropout=dropout)
        self.norm1 = nn.LayerNorm(dim, elementwise_affine=False, eps=1e-6)
        self.norm2 = nn.LayerNorm(dim, elementwise_affine=False, eps=1e-6)
        self.adaLN_modulation = nn.Sequential(
            nn.SiLU(), nn.Linear(dim, 6 * dim, bias=True)
        )

        nn.init.constant_(self.adaLN_modulation[-1].weight, 0)
        nn.init.constant_(self.adaLN_modulation[-1].bias, 0)

    def forward(self, x, c, attn_mask=None):
        shift_msa, scale_msa, gate_msa, shift_mlp, scale_mlp, gate_mlp = (
            self.adaLN_modulation(c).chunk(6, dim=-1)
        )
        x = x + gate_msa * self.attn(
            modulate(self.norm1(x), shift_msa, scale_msa),
            causal=attn_mask is None,
            attn_mask=attn_mask,
        )
        x = x + gate_mlp * self.mlp(modulate(self.norm2(x), shift_mlp, scale_mlp))
        return x


class Block(nn.Module):
    """Standard Transformer block"""

    def __init__(self, dim, heads, dim_head, mlp_dim, dropout=0.0):
        super().__init__()

        self.attn = Attention(dim, heads=heads, dim_head=dim_head, dropout=dropout)
        self.mlp = FeedForward(dim, mlp_dim, dropout=dropout)
        self.norm1 = nn.LayerNorm(dim, elementwise_affine=False, eps=1e-6)
        self.norm2 = nn.LayerNorm(dim, elementwise_affine=False, eps=1e-6)

    def forward(self, x, attn_mask=None):
        x = x + self.attn(
            self.norm1(x), causal=attn_mask is None, attn_mask=attn_mask
        )
        x = x + self.mlp(self.norm2(x))
        return x


class Transformer(nn.Module):
    """Standard Transformer with support for AdaLN-zero blocks"""

    def __init__(
        self,
        input_dim,
        hidden_dim,
        output_dim,
        depth,
        heads,
        dim_head,
        mlp_dim,
        dropout=0.0,
        block_class=Block,
    ):
        super().__init__()
        self.norm = nn.LayerNorm(hidden_dim)
        self.layers = nn.ModuleList([])

        self.input_proj = (
            nn.Linear(input_dim, hidden_dim)
            if input_dim != hidden_dim
            else nn.Identity()
        )

        self.cond_proj = (
            nn.Linear(input_dim, hidden_dim)
            if input_dim != hidden_dim
            else nn.Identity()
        )

        self.output_proj = (
            nn.Linear(hidden_dim, output_dim)
            if hidden_dim != output_dim
            else nn.Identity()
        )

        for _ in range(depth):
            self.layers.append(
                block_class(hidden_dim, heads, dim_head, mlp_dim, dropout)
            )

    def forward(self, x, c=None, attn_mask=None, pe=None, inject_every_layer=False):

        if hasattr(self, "input_proj"):
            x = self.input_proj(x)
            if pe is not None:
                pe = self.input_proj(pe)

        if c is not None and hasattr(self, "cond_proj"):
            c = self.cond_proj(c)

        if pe is not None and not inject_every_layer:
            x = x + pe

        for block in self.layers:
            if pe is not None and inject_every_layer:
                x = x + pe
            if isinstance(block, Block):
                x = block(x, attn_mask=attn_mask)
            else:
                x = block(x, c, attn_mask=attn_mask)
        x = self.norm(x)

        if hasattr(self, "output_proj"):
            x = self.output_proj(x)
        return x

class Embedder(nn.Module):
    def __init__(
        self,
        input_dim=10,
        smoothed_dim=10,
        emb_dim=10,
        mlp_scale=4,
    ):
        super().__init__()
        self.patch_embed = nn.Conv1d(input_dim, smoothed_dim, kernel_size=1, stride=1)
        self.embed = nn.Sequential(
            nn.Linear(smoothed_dim, mlp_scale * emb_dim),
            nn.SiLU(),
            nn.Linear(mlp_scale * emb_dim, emb_dim),
        )

    def forward(self, x):
        """
        x: (B, T, D)
        """
        x = x.float()
        x = x.permute(0, 2, 1)
        x = self.patch_embed(x)
        x = x.permute(0, 2, 1)
        x = self.embed(x)
        return x


class MLP(nn.Module):
    """Simple MLP with optional normalization and activation"""

    def __init__(
        self,
        input_dim,
        hidden_dim,
        output_dim=None,
        norm_fn=nn.LayerNorm,
        act_fn=nn.GELU,
    ):
        super().__init__()
        norm_fn = norm_fn(hidden_dim) if norm_fn is not None else nn.Identity()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            norm_fn,
            act_fn(),
            nn.Linear(hidden_dim, output_dim or input_dim),
        )

    def forward(self, x):
        """
        x: (B*T, D)
        """
        return self.net(x)


def block_causal_attn_mask(num_frames: int, num_spatial: int, device, dtype):
    """Additive mask: attend to all spatial tokens in frames <= current frame."""
    t_idx = torch.arange(num_frames, device=device).repeat_interleave(num_spatial)
    allow = t_idx[None, :] <= t_idx[:, None]
    mask = torch.zeros(t_idx.numel(), t_idx.numel(), device=device, dtype=dtype)
    return mask.masked_fill(~allow, torch.finfo(dtype).min)


class NibblePE(nn.Module):
    """Tokenization-aligned two-scale (radix-16) position encoding.

    High nibble = 4x4 slot id, low nibble = 4x4 offset inside the slot.
    Both scales share the same 4x4 row/col embeddings.
    """

    def __init__(self, dim: int, spatial_grid: int = 4, include_global: bool = True):
        super().__init__()
        self.dim = int(dim)
        self.spatial_grid = int(spatial_grid)
        g = self.spatial_grid
        self.row_embed = nn.Embedding(g, dim)
        self.col_embed = nn.Embedding(g, dim)
        self.mlp_coarse = nn.Sequential(nn.Linear(dim, dim), nn.GELU(), nn.Linear(dim, dim))
        self.mlp_fine = nn.Sequential(nn.Linear(dim, dim), nn.GELU(), nn.Linear(dim, dim))
        self.include_global = bool(include_global)
        self.global_embed = (
            nn.Parameter(torch.zeros(1, 1, 1, dim)) if self.include_global else None
        )
        rel = 2 * (g - 1) + 1
        self.rel_span = g - 1
        self.b_coarse = nn.Parameter(torch.zeros(rel, rel))
        self.b_fine = nn.Parameter(torch.zeros(rel, rel))

    def cell_embed(self, row: torch.Tensor, col: torch.Tensor) -> torch.Tensor:
        return self.row_embed(row) + self.col_embed(col)

    def absolute(self, n_hi: torch.Tensor, n_lo: torch.Tensor) -> torch.Tensor:
        g = self.spatial_grid
        row_h, col_h = n_hi // g, n_hi % g
        row_l, col_l = n_lo // g, n_lo % g
        return self.mlp_coarse(self.cell_embed(row_h, col_h)) + self.mlp_fine(
            self.cell_embed(row_l, col_l)
        )

    def relative_bias(self, n_hi: torch.Tensor, n_lo: torch.Tensor, num_tokens: int):
        """Additive spatial bias (B, TN, TN). Global token (index 0) gets 0.

        Coarse Δ(R,C) always applies between slots. Fine Δ(r,c) only when
        n_hi matches (same 4x4 cell); cross-slot fine term is gated to 0.
        Never treat the packed byte as a 1D distance.
        """
        B, T, S = n_hi.shape
        g = self.spatial_grid
        N = int(num_tokens)
        include_global = N == S + 1
        device = n_hi.device
        slot = slice(1, None) if include_global else slice(None)

        # (B, T, N) grids; global positions stay -1 and are zeroed via ``valid``.
        r_hi = torch.full((B, T, N), -1, device=device, dtype=torch.long)
        c_hi = r_hi.clone()
        r_lo = r_hi.clone()
        c_lo = r_hi.clone()
        hi_id = r_hi.clone()
        r_hi[:, :, slot] = n_hi // g
        c_hi[:, :, slot] = n_hi % g
        r_lo[:, :, slot] = n_lo // g
        c_lo[:, :, slot] = n_lo % g
        hi_id[:, :, slot] = n_hi

        def _pair_delta(coord):
            flat = coord.reshape(B, T * N)
            return flat[:, :, None] - flat[:, None, :]

        flat_hi = hi_id.reshape(B, T * N)
        valid = (flat_hi[:, :, None] >= 0) & (flat_hi[:, None, :] >= 0)
        same_cell = (flat_hi[:, :, None] == flat_hi[:, None, :]) & valid

        span = self.rel_span

        def _lookup(table, dr, dc):
            rr = dr.clamp(-span, span) + span
            cc = dc.clamp(-span, span) + span
            return table[rr, cc]

        coarse = _lookup(self.b_coarse, _pair_delta(r_hi), _pair_delta(c_hi))
        fine = _lookup(self.b_fine, _pair_delta(r_lo), _pair_delta(c_lo))
        bias = coarse + same_cell.to(dtype=fine.dtype) * fine
        return torch.where(valid, bias, torch.zeros_like(bias))


class ARPredictor(nn.Module):
    """Autoregressive predictor for next-step embedding prediction.

    Supports CLS (N=1, sequence length T) and spatial tokens (N>1, length T*N)
    with time-block causal attention.
    """

    def __init__(
        self,
        *,
        num_frames,
        depth,
        heads,
        mlp_dim,
        input_dim,
        hidden_dim,
        output_dim=None,
        dim_head=64,
        dropout=0.0,
        emb_dropout=0.0,
        num_spatial_tokens: int = 1,
        pos_mode: str = "learned",
        include_global_token: bool = False,
        inject_every_layer: bool = False,
        n_lo_mode: str = "attn",
        spatial_grid: int = 4,
    ):
        super().__init__()
        self.num_frames = num_frames
        self.num_spatial_tokens = int(num_spatial_tokens)
        self.pos_mode = str(pos_mode)
        self.include_global_token = bool(include_global_token)
        self.inject_every_layer = bool(inject_every_layer)
        self.n_lo_mode = str(n_lo_mode)
        self.spatial_grid = int(spatial_grid)
        self.pos_embedding = nn.Parameter(torch.randn(1, num_frames, input_dim))
        use_learned_spatial = self.num_spatial_tokens > 1 and self.pos_mode != "nibble"
        if use_learned_spatial:
            self.spatial_pos_embedding = nn.Parameter(
                torch.randn(1, 1, self.num_spatial_tokens, input_dim)
            )
        else:
            self.spatial_pos_embedding = None
        self.nibble_pe = None
        if self.pos_mode == "nibble" and self.num_spatial_tokens > 1:
            self.nibble_pe = NibblePE(
                dim=input_dim,
                spatial_grid=self.spatial_grid,
                include_global=self.include_global_token,
            )
        self.dropout = nn.Dropout(emb_dropout)
        self.transformer = Transformer(
            input_dim,
            hidden_dim,
            output_dim or input_dim,
            depth,
            heads,
            dim_head,
            mlp_dim,
            dropout,
            block_class=ConditionalBlock,
        )

    def _fallback_nibble(self, batch, time, slots, device):
        g = self.spatial_grid
        n_hi = torch.arange(slots, device=device).view(1, 1, slots).expand(batch, time, -1)
        center = (g // 2) * g + (g // 2)
        n_lo = torch.full((batch, time, slots), center, device=device, dtype=torch.long)
        return n_hi, n_lo

    def forward(self, x, c, n_hi=None, n_lo=None):
        """
        x: (B, T, d) or (B, T, N, d)
        c: (B, T, act_dim)
        """
        spatial = x.ndim == 4
        if spatial:
            B, T, N, D = x.shape
            assert N == self.num_spatial_tokens, (
                f"expected {self.num_spatial_tokens} spatial tokens, got {N}"
            )
            x = x + self.pos_embedding[:, :T, None, :]
            extra_pe = None
            attn_mask = block_causal_attn_mask(T, N, x.device, x.dtype)
            pos_mode = getattr(self, "pos_mode", "learned")
            nibble_pe = getattr(self, "nibble_pe", None)
            include_global = getattr(self, "include_global_token", False)
            inject_every_layer = getattr(self, "inject_every_layer", False)
            if pos_mode == "nibble" and nibble_pe is not None:
                n_slots = N - int(include_global)
                n_lo_mode = getattr(self, "n_lo_mode", "attn")
                if n_lo_mode == "fixed_center" or n_hi is None or n_lo is None:
                    n_hi, n_lo = self._fallback_nibble(B, T, n_slots, x.device)
                # discrete codes are indices only; keep them off the autograd graph
                n_hi = n_hi.long().detach()
                n_lo = n_lo.long().detach()
                pe_slots = nibble_pe.absolute(n_hi, n_lo)
                if include_global:
                    gpe = nibble_pe.global_embed.expand(B, T, 1, D)
                    extra_pe = torch.cat([gpe, pe_slots], dim=2)
                else:
                    extra_pe = pe_slots
                rel = nibble_pe.relative_bias(n_hi, n_lo, num_tokens=N)
                # (TN,TN) causal + (B,TN,TN) relative -> (B,1,TN,TN) for SDPA
                attn_mask = attn_mask.to(dtype=x.dtype)
                attn_mask = attn_mask.unsqueeze(0) + rel.to(dtype=x.dtype)
                attn_mask = attn_mask.unsqueeze(1)
            elif self.spatial_pos_embedding is not None:
                x = x + self.spatial_pos_embedding[:, :, :N, :]

            c = c.unsqueeze(2).expand(-1, -1, N, -1)
            x = rearrange(x, "b t n d -> b (t n) d")
            c = rearrange(c, "b t n d -> b (t n) d")
            if extra_pe is not None:
                extra_pe = rearrange(extra_pe, "b t n d -> b (t n) d")
            x = self.dropout(x)
            if extra_pe is not None:
                extra_pe = self.dropout(extra_pe)
            x = self.transformer(
                x,
                c,
                attn_mask=attn_mask,
                pe=extra_pe,
                inject_every_layer=inject_every_layer and extra_pe is not None,
            )
            return rearrange(x, "b (t n) d -> b t n d", t=T, n=N)

        T = x.size(1)
        x = x + self.pos_embedding[:, :T]
        x = self.dropout(x)
        x = self.transformer(x, c)
        return x
