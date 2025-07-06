"""
Bidirectional Mamba2 implementation based on nd_mamba2.py
Using the dual forward/backward Mamba2 architecture for bidirectional processing
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from dataclasses import dataclass
from typing import NamedTuple, Optional
from einops import rearrange, repeat

Device = torch.device


@dataclass
class Mamba2Config:
    d_model: int  # model dimension (D)
    n_layer: int = 24  # number of Mamba-2 layers in the language model
    d_state: int = 128  # state dimension (N)
    d_conv: int = 4  # convolution kernel size
    expand: int = 2  # expansion factor (E)
    headdim: int = 64  # head dimension (P)
    chunk_size: int = 64  # matrix partition size (Q)
    vocab_size: int = 50277
    pad_vocab_size_multiple: int = 16

    def __post_init__(self):
        self.d_inner = self.expand * self.d_model
        assert self.d_inner % self.headdim == 0
        self.nheads = self.d_inner // self.headdim
        if self.vocab_size % self.pad_vocab_size_multiple != 0:
            self.vocab_size += (
                    self.pad_vocab_size_multiple
                    - self.vocab_size % self.pad_vocab_size_multiple
            )


class InferenceCache(NamedTuple):
    conv_state: torch.Tensor  # (batch, d_inner + 2 * d_state, d_conv)
    ssm_state: torch.Tensor  # (batch, nheads, headdim, d_state)

    @staticmethod
    def alloc(batch_size: int, args: Mamba2Config, device: Device = None):
        return InferenceCache(
            torch.zeros(
                batch_size, args.d_inner + 2 * args.d_state, args.d_conv, device=device
            ),
            torch.zeros(
                batch_size, args.nheads, args.headdim, args.d_state, device=device
            ),
        )


def segsum(x: torch.Tensor, device: Device = None) -> torch.Tensor:
    """Stable segment sum calculation."""
    T = x.size(-1)
    x = repeat(x, "... d -> ... d e", e=T)
    mask = torch.tril(torch.ones(T, T, dtype=torch.bool, device=device), diagonal=-1)
    x = x.masked_fill(~mask, 0)
    x_segsum = torch.cumsum(x, dim=-2)
    mask = torch.tril(torch.ones(T, T, dtype=torch.bool, device=device), diagonal=0)
    x_segsum = x_segsum.masked_fill(~mask, -1e9)
    return x_segsum


def ssd(x, A, B, C, chunk_size, initial_states=None, device: Device = None):
    """Structured State Space Duality (SSD) - the core of Mamba-2"""
    assert x.shape[1] % chunk_size == 0

    # Rearrange into chunks
    x, A, B, C = [
        rearrange(m, "b (c l) ... -> b c l ...", l=chunk_size) for m in (x, A, B, C)
    ]

    A = rearrange(A, "b c l h -> b h c l")
    A_cumsum = torch.cumsum(A, dim=-1)

    # 1. Compute the output for each intra-chunk (diagonal blocks)
    L = torch.exp(torch.clamp(segsum(A, device=device), min=-10, max=10))
    Y_diag = torch.einsum("bclhn, bcshn, bhcls, bcshp -> bclhp", C, B, L, x)

    # 2. Compute the state for each intra-chunk
    decay_states = torch.exp(torch.clamp(A_cumsum[:, :, :, -1:] - A_cumsum, min=-10, max=10))
    states = torch.einsum("bclhn, bhcl, bclhp -> bchpn", B, decay_states, x)

    # 3. Compute the inter-chunk SSM recurrence
    if initial_states is None:
        initial_states = torch.zeros_like(states[:, :1])
    states = torch.cat([initial_states, states], dim=1)
    decay_chunk = torch.exp(torch.clamp(segsum(F.pad(A_cumsum[:, :, :, -1], (1, 0)), device=device), min=-10, max=10))
    new_states = torch.einsum("bhzc, bchpn -> bzhpn", decay_chunk, states)
    states, final_state = new_states[:, :-1], new_states[:, -1]

    # 4. Compute state -> output conversion per chunk
    state_decay_out = torch.exp(torch.clamp(A_cumsum, min=-10, max=10))
    Y_off = torch.einsum("bclhn, bchpn, bhcl -> bclhp", C, states, state_decay_out)

    # Add output of intra-chunk and inter-chunk terms
    Y = rearrange(Y_diag + Y_off, "b c l h p -> b (c l) h p")

    return Y, final_state


class RMSNorm(nn.Module):
    def __init__(self, d: int, eps: float = 1e-5, device: Device = None):
        """Gated Root Mean Square Layer Normalization"""
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(d, device=device))

    def forward(self, x, z=None):
        if z is not None:
            x = x * silu(z)
        return x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps) * self.weight


def silu(x):
    """Applies the Sigmoid Linear Unit (SiLU), element-wise."""
    return x * F.sigmoid(x)


class Mamba2(nn.Module):
    def __init__(self, d_model: int, n_layer: int = 24, d_state: int = 128, 
                 d_conv: int = 4, expand: int = 2, headdim: int = 64, 
                 chunk_size: int = 64, vocab_size: int = 50277, 
                 pad_vocab_size_multiple: int = 16, bias: bool = False, 
                 conv_bias: bool = True):
        super().__init__()
        args = Mamba2Config(d_model, n_layer, d_state, d_conv, expand, headdim, 
                           chunk_size, vocab_size, pad_vocab_size_multiple)
        self.args = args
        
        # Order: (z, x, B, C, dt)
        d_in_proj = 2 * args.d_inner + 2 * args.d_state + args.nheads
        self.in_proj = nn.Linear(args.d_model, d_in_proj, bias=bias)

        conv_dim = args.d_inner + 2 * args.d_state
        self.conv1d = nn.Conv1d(
            in_channels=conv_dim,
            out_channels=conv_dim,
            kernel_size=args.d_conv,
            groups=conv_dim,
            padding=args.d_conv - 1,
            bias=conv_bias,
        )

        self.dt_bias = nn.Parameter(torch.empty(args.nheads, ))
        self.A_log = nn.Parameter(torch.empty(args.nheads, ))
        self.D = nn.Parameter(torch.empty(args.nheads, ))
        self.norm = RMSNorm(args.d_inner, )
        self.out_proj = nn.Linear(args.d_inner, args.d_model, bias=bias)
        
        # Initialize parameters properly
        self._init_parameters()
        
    def _init_parameters(self):
        """Initialize parameters for numerical stability"""
        # Initialize dt_bias
        nn.init.uniform_(self.dt_bias, -0.1, 0.1)
        
        # Initialize A_log (should be negative for stability)
        nn.init.uniform_(self.A_log, -2.0, -1.0)
        
        # Initialize D
        nn.init.uniform_(self.D, -0.1, 0.1)
        
        # Initialize linear layers
        nn.init.xavier_uniform_(self.in_proj.weight, gain=0.5)
        nn.init.xavier_uniform_(self.out_proj.weight, gain=0.5)
        
        # Initialize conv layer
        nn.init.xavier_uniform_(self.conv1d.weight, gain=1.0)
        if self.conv1d.bias is not None:
            nn.init.zeros_(self.conv1d.bias)

    def forward(self, u: torch.Tensor, h=None):
        """
        Arguments
            u: (batch, seqlen, d_model) input. seqlen should be a multiple of chunk_size.
            h: hidden states for inference step. Initialized to 0s if not present.

        Return (y, h)
            y: (batch, seqlen, d_model) output
            h: updated inference cache after processing `u`
        """
        if h:
            return self.step(u, h)

        A = -torch.exp(self.A_log)  # (nheads,)
        zxbcdt = self.in_proj(u)  # (batch, seqlen, d_in_proj)
        z, xBC, dt = torch.split(
            zxbcdt,
            [
                self.args.d_inner,
                self.args.d_inner + 2 * self.args.d_state,
                self.args.nheads,
            ],
            dim=-1,
        )
        dt = F.softplus(dt + self.dt_bias)  # (batch, seqlen, nheads)

        # Pad or truncate xBC seqlen to d_conv
        conv_state = F.pad(
            rearrange(xBC, "b l d -> b d l"), (self.args.d_conv - u.shape[1], 0)
        )

        xBC = silu(
            self.conv1d(xBC.transpose(1, 2)).transpose(1, 2)[:, : u.shape[1], :]
        )  # (batch, seqlen, d_inner + 2 * d_state))
        x, B, C = torch.split(
            xBC, [self.args.d_inner, self.args.d_state, self.args.d_state], dim=-1
        )
        x = rearrange(x, "b l (h p) -> b l h p", p=self.args.headdim)
        y, ssm_state = ssd(
            x * dt.unsqueeze(-1),
            A * dt,
            rearrange(B, "b l n -> b l 1 n"),
            rearrange(C, "b l n -> b l 1 n"),
            self.args.chunk_size,
            device=x.device,
        )
        y = y + x * self.D.unsqueeze(-1)
        y = rearrange(y, "b l h p -> b l (h p)")
        y = self.norm(y, z)
        y = self.out_proj(y)

        h = InferenceCache(conv_state, ssm_state)
        return y, h

    def step(self, u: torch.Tensor, h: InferenceCache):
        """Take a single inference step for the current input and hidden state"""
        assert u.shape[1] == 1, "Only one token can be decoded per inference step"

        zxbcdt = self.in_proj(u.squeeze(1))  # (batch, d_in_proj)
        z, xBC, dt = torch.split(
            zxbcdt,
            [
                self.args.d_inner,
                self.args.d_inner + 2 * self.args.d_state,
                self.args.nheads,
            ],
            dim=-1,
        )

        # Advance convolution input
        h.conv_state.copy_(torch.roll(h.conv_state, shifts=-1, dims=-1))
        h.conv_state[:, :, -1] = xBC
        # Convolution step
        xBC = torch.sum(
            h.conv_state * rearrange(self.conv1d.weight, "d 1 w -> d w"), dim=-1
        )
        xBC += self.conv1d.bias
        xBC = silu(xBC)

        x, B, C = torch.split(
            xBC, [self.args.d_inner, self.args.d_state, self.args.d_state], dim=-1
        )
        A = -torch.exp(self.A_log)  # (nheads,)

        # SSM step
        dt = F.softplus(dt + self.dt_bias)  # (batch, nheads)
        dA = torch.exp(dt * A)  # (batch, nheads)
        x = rearrange(x, "b (h p) -> b h p", p=self.args.headdim)
        dBx = torch.einsum("bh, bn, bhp -> bhpn", dt, B, x)
        h.ssm_state.copy_(h.ssm_state * rearrange(dA, "b h -> b h 1 1") + dBx)
        y = torch.einsum("bhpn, bn -> bhp", h.ssm_state, C)
        y = y + rearrange(self.D, "h -> h 1") * x
        y = rearrange(y, "b h p -> b (h p)")
        y = self.norm(y, z)
        y = self.out_proj(y)

        return y.unsqueeze(1), h


class BidirectionalMamba2(nn.Module):
    """Bidirectional Mamba2 block using forward and backward Mamba2 layers"""
    def __init__(self, d_model: int, d_state: int = 64, d_conv: int = 4, 
                 expand: int = 2, headdim: int = 32, chunk_size: int = 64, 
                 dropout: float = 0.1, bias: bool = False, conv_bias: bool = True):
        super().__init__()
        
        # Forward and backward Mamba2 layers
        self.mamba2_for = Mamba2(
            d_model=d_model, 
            d_state=d_state, 
            d_conv=d_conv, 
            expand=expand, 
            headdim=headdim, 
            chunk_size=chunk_size,
            bias=bias,
            conv_bias=conv_bias
        )
        self.mamba2_back = Mamba2(
            d_model=d_model, 
            d_state=d_state, 
            d_conv=d_conv, 
            expand=expand, 
            headdim=headdim, 
            chunk_size=chunk_size,
            bias=bias,
            conv_bias=conv_bias
        )
        
        self.dropout = nn.Dropout(dropout)
        self.layer_norm = nn.LayerNorm(d_model)

    def forward(self, x):
        """
        x: (batch, seqlen, d_model)
        """
        # Ensure sequence length is compatible with chunk_size
        batch_size, seq_len, d_model = x.shape
        chunk_size = self.mamba2_for.args.chunk_size
        
        # Pad sequence to be multiple of chunk_size
        pad_len = (chunk_size - seq_len % chunk_size) % chunk_size
        if pad_len > 0:
            x = F.pad(x, (0, 0, 0, pad_len))
        
        # Forward direction
        x1, h1 = self.mamba2_for(x)
        
        # Backward direction
        x2, h2 = self.mamba2_back(x.flip(1))
        x2 = x2.flip(1)
        
        # Combine forward and backward outputs
        y = x1 + x2
        
        # Truncate back to original length
        if pad_len > 0:
            y = y[:, :seq_len, :]
        
        # Apply dropout and residual connection
        y = self.dropout(y)
        y = self.layer_norm(y + x[:, :seq_len, :])
        
        return y


class BMambaBlocks(nn.Module):
    """Stack of Bidirectional Mamba2 blocks, compatible with ConformerBlocks interface"""
    def __init__(
        self,
        encoder_dim: int = 512,
        num_layers: int = 2,
        d_state: int = 128,
        d_conv: int = 4,
        expand: int = 2,
        headdim: int = 64,
        chunk_size: int = 64,
        dropout: float = 0.1,
        bias: bool = False,
        conv_bias: bool = True,
        **kwargs  # For compatibility with other arguments
    ):
        super().__init__()
        
        # Handle alternative parameter names for compatibility
        if 'bmamba_d_state' in kwargs:
            d_state = kwargs['bmamba_d_state']
        if 'bmamba_d_conv' in kwargs:
            d_conv = kwargs['bmamba_d_conv']
        if 'bmamba_expand' in kwargs:
            expand = kwargs['bmamba_expand']
        if 'bmamba_dropout' in kwargs:
            dropout = kwargs['bmamba_dropout']
        if 'bmamba_bias' in kwargs:
            bias = kwargs['bmamba_bias']
        if 'bmamba_conv_bias' in kwargs:
            conv_bias = kwargs['bmamba_conv_bias']
        
        # Ensure headdim compatibility
        d_inner = expand * encoder_dim
        if d_inner % headdim != 0:
            # Find largest compatible headdim
            headdim = 64  # Default safe value
            while d_inner % headdim != 0 and headdim > 1:
                headdim //= 2
            if headdim < 1:
                headdim = 1
            print(f"Warning: Adjusted headdim to {headdim} for compatibility (d_inner={d_inner})")
        
        # Ensure chunk_size is compatible with sequence processing
        if chunk_size > 256:  # Avoid memory issues
            chunk_size = 64
            print(f"Warning: Reduced chunk_size to {chunk_size} for stability")
        
        self.layers = nn.ModuleList([
            BidirectionalMamba2(
                d_model=encoder_dim,
                d_state=d_state,
                d_conv=d_conv,
                expand=expand,
                headdim=headdim,
                chunk_size=chunk_size,
                dropout=dropout,
                bias=bias,
                conv_bias=conv_bias,
            )
            for _ in range(num_layers)
        ])
        
        self.norm = nn.LayerNorm(encoder_dim)
        
        # Print configuration for debugging
        print(f"BMambaBlocks initialized:")
        print(f"  encoder_dim: {encoder_dim}, num_layers: {num_layers}")
        print(f"  d_state: {d_state}, d_conv: {d_conv}, expand: {expand}")
        print(f"  headdim: {headdim}, chunk_size: {chunk_size}, dropout: {dropout}")
        print(f"  bias: {bias}, conv_bias: {conv_bias}")
        
    def forward(self, inputs):
        """
        inputs: (batch, seq_len, d_model)
        Returns: (batch, seq_len, d_model)
        """
        x = inputs
        for layer in self.layers:
            x = layer(x)
        
        return self.norm(x)


# For backward compatibility, keep SimpleBiMamba as an alternative
class SimpleBiMamba(nn.Module):
    """Simplified bidirectional processing using LSTM for fallback"""
    def __init__(
        self,
        d_model: int,
        hidden_dim: Optional[int] = None,
        num_layers: int = 2,
        dropout: float = 0.1,
        **kwargs
    ):
        super().__init__()
        
        if hidden_dim is None:
            hidden_dim = d_model * 2
            
        # Bidirectional LSTM as fallback
        self.lstm = nn.LSTM(
            input_size=d_model,
            hidden_size=hidden_dim // 2,
            num_layers=num_layers,
            dropout=dropout if num_layers > 1 else 0,
            bidirectional=True,
            batch_first=True
        )
        
        self.output_proj = nn.Linear(hidden_dim, d_model)
        self.layer_norm = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, inputs):
        """
        inputs: (batch, seq_len, d_model)
        """
        lstm_out, _ = self.lstm(inputs)
        output = self.output_proj(lstm_out)
        output = self.dropout(output)
        
        # Residual connection and normalization
        output = self.layer_norm(output + inputs)
        
        return output 