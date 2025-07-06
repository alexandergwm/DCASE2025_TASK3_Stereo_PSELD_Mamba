import torch
from torch import Tensor, nn
from torch.nn import functional as F
from abc import abstractmethod
import math


def silu(x):
    return x * F.sigmoid(x)


class RMSNorm(nn.Module):
    def __init__(self, d: int, eps: float = 1e-5):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(d))

    def forward(self, x, z):
        x = x * silu(z)
        return x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps) * self.weight


class Mamba2(nn.Module):
    def __init__(self, d_model: int,  # model dimension (D)
                 n_layer: int = 24,  # number of Mamba-2 layers in the language model
                 d_state: int = 128,  # state dimension (N)
                 d_conv: int = 4,  # convolution kernel size
                 expand: int = 2,  # expansion factor (E)
                 headdim: int = 64,  # head dimension (P)
                 chunk_size: int = 64,  # matrix partition size (Q)
                 ):
        super().__init__()
        self.n_layer = n_layer
        self.d_state = d_state
        self.headdim = headdim
        # self.chunk_size = torch.tensor(chunk_size, dtype=torch.int32)
        self.chunk_size = chunk_size

        self.d_inner = expand * d_model
        assert self.d_inner % self.headdim == 0, "self.d_inner must be divisible by self.headdim"
        self.nheads = self.d_inner // self.headdim

        d_in_proj = 2 * self.d_inner + 2 * self.d_state + self.nheads
        self.in_proj = nn.Linear(d_model, d_in_proj, bias=False)

        conv_dim = self.d_inner + 2 * d_state
        self.conv1d = nn.Conv1d(conv_dim, conv_dim, d_conv, groups=conv_dim, padding=d_conv - 1, )
        self.dt_bias = nn.Parameter(torch.empty(self.nheads, ))
        self.A_log = nn.Parameter(torch.empty(self.nheads, ))
        self.D = nn.Parameter(torch.empty(self.nheads, ))
        self.norm = RMSNorm(self.d_inner, )
        self.out_proj = nn.Linear(self.d_inner, d_model, bias=False, )
        
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

    def forward(self, u: Tensor):
        A = -torch.exp(self.A_log)  # (nheads,)
        zxbcdt = self.in_proj(u)  # (batch, seqlen, d_in_proj)
        z, xBC, dt = torch.split(
            zxbcdt,
            [
                self.d_inner,
                self.d_inner + 2 * self.d_state,
                self.nheads,
            ],
            dim=-1,
        )
        dt = F.softplus(dt + self.dt_bias)  # (batch, seqlen, nheads)

        # Pad or truncate xBC seqlen to d_conv
        xBC = silu(
            self.conv1d(xBC.transpose(1, 2)).transpose(1, 2)[:, : u.shape[1], :]
        )  # (batch, seqlen, d_inner + 2 * d_state))
        x, B, C = torch.split(
            xBC, [self.d_inner, self.d_state, self.d_state], dim=-1
        )

        _b, _l, _hp = x.shape
        _h = _hp // self.headdim
        _p = self.headdim
        x = x.reshape(_b, _l, _h, _p)

        y = self.ssd(x * dt.unsqueeze(-1),
                     A * dt,
                     B.unsqueeze(2),
                     C.unsqueeze(2), )

        y = y + x * self.D.unsqueeze(-1)

        _b, _l, _h, _p = y.shape
        y = y.reshape(_b, _l, _h * _p)

        y = self.norm(y, z)
        y = self.out_proj(y)

        return y

    def segsum(self, x: Tensor) -> Tensor:
        T = x.size(-1)
        device = x.device
        x = x[..., None].repeat(1, 1, 1, 1, T)
        mask = torch.tril(torch.ones(T, T, dtype=torch.bool, device=device), diagonal=-1)
        x = x.masked_fill(~mask, 0)
        x_segsum = torch.cumsum(x, dim=-2)
        mask = torch.tril(torch.ones(T, T, dtype=torch.bool, device=device), diagonal=0)
        x_segsum = x_segsum.masked_fill(~mask, -torch.inf)
        return x_segsum

    def ssd(self, x, A, B, C):
        chunk_size = self.chunk_size
        # if x.shape[1] % chunk_size == 0:
        #
        x = x.reshape(x.shape[0], x.shape[1] // chunk_size, chunk_size, x.shape[2], x.shape[3], )
        B = B.reshape(B.shape[0], B.shape[1] // chunk_size, chunk_size, B.shape[2], B.shape[3], )
        C = C.reshape(C.shape[0], C.shape[1] // chunk_size, chunk_size, C.shape[2], C.shape[3], )
        A = A.reshape(A.shape[0], A.shape[1] // chunk_size, chunk_size, A.shape[2])
        A = A.permute(0, 3, 1, 2)
        A_cumsum = torch.cumsum(A, dim=-1)

        # 1. Compute the output for each intra-chunk (diagonal blocks)
        L = torch.exp(self.segsum(A))
        Y_diag = torch.einsum("bclhn, bcshn, bhcls, bcshp -> bclhp", C, B, L, x)

        # 2. Compute the state for each intra-chunk
        # (right term of low-rank factorization of off-diagonal blocks; B terms)
        decay_states = torch.exp(A_cumsum[:, :, :, -1:] - A_cumsum)
        states = torch.einsum("bclhn, bhcl, bclhp -> bchpn", B, decay_states, x)

        # 3. Compute the inter-chunk SSM recurrence; produces correct SSM states at chunk boundaries
        # (middle term of factorization of off-diag blocks; A terms)

        initial_states = torch.zeros_like(states[:, :1])
        states = torch.cat([initial_states, states], dim=1)

        decay_chunk = torch.exp(self.segsum(F.pad(A_cumsum[:, :, :, -1], (1, 0))))[0]
        new_states = torch.einsum("bhzc, bchpn -> bzhpn", decay_chunk, states)
        states = new_states[:, :-1]

        # 4. Compute state -> output conversion per chunk
        # (left term of low-rank factorization of off-diagonal blocks; C terms)
        state_decay_out = torch.exp(A_cumsum)
        Y_off = torch.einsum("bclhn, bchpn, bhcl -> bclhp", C, states, state_decay_out)

        # Add output of intra-chunk and inter-chunk terms (diagonal and off-diagonal blocks)
        # Y = rearrange(Y_diag + Y_off, "b c l h p -> b (c l) h p")
        Y = Y_diag + Y_off
        Y = Y.reshape(Y.shape[0], Y.shape[1] * Y.shape[2], Y.shape[3], Y.shape[4], )

        return Y


class BidirectionalMamba2_v2(nn.Module):
    """Bidirectional Mamba2 block using forward and backward Mamba2 layers"""
    def __init__(self, d_model: int, d_state: int = 128, d_conv: int = 4, 
                 expand: int = 2, headdim: int = 64, chunk_size: int = 64, 
                 dropout: float = 0.1):
        super().__init__()
        
        # Forward and backward Mamba2 layers
        self.mamba2_for = Mamba2(
            d_model=d_model, 
            d_state=d_state, 
            d_conv=d_conv, 
            expand=expand, 
            headdim=headdim, 
            chunk_size=chunk_size
        )
        self.mamba2_back = Mamba2(
            d_model=d_model, 
            d_state=d_state, 
            d_conv=d_conv, 
            expand=expand, 
            headdim=headdim, 
            chunk_size=chunk_size
        )
        
        self.dropout = nn.Dropout(dropout)
        self.layer_norm = nn.LayerNorm(d_model)
        self.chunk_size = chunk_size

    def forward(self, x):
        """
        x: (batch, seqlen, d_model)
        """
        # Ensure sequence length is compatible with chunk_size
        batch_size, seq_len, d_model = x.shape
        
        # Pad sequence to be multiple of chunk_size
        pad_len = (self.chunk_size - seq_len % self.chunk_size) % self.chunk_size
        if pad_len > 0:
            x = F.pad(x, (0, 0, 0, pad_len))
        
        # Forward direction
        x1 = self.mamba2_for(x)
        
        # Backward direction
        x2 = self.mamba2_back(x.flip(1))
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


class BiMambaBlocks2(nn.Module):
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
        
        # Ensure chunk_size is reasonable
        if chunk_size > 256:  # Avoid memory issues
            chunk_size = 64
            print(f"Warning: Reduced chunk_size to {chunk_size} for stability")
        
        self.layers = nn.ModuleList([
            BidirectionalMamba2_v2(
                d_model=encoder_dim,
                d_state=d_state,
                d_conv=d_conv,
                expand=expand,
                headdim=headdim,
                chunk_size=chunk_size,
                dropout=dropout,
            )
            for _ in range(num_layers)
        ])
        
        self.norm = nn.LayerNorm(encoder_dim)
        
        # Print configuration for debugging
        print(f"BiMambaBlocks2 initialized:")
        print(f"  encoder_dim: {encoder_dim}, num_layers: {num_layers}")
        print(f"  d_state: {d_state}, d_conv: {d_conv}, expand: {expand}")
        print(f"  headdim: {headdim}, chunk_size: {chunk_size}, dropout: {dropout}")
        
    def forward(self, inputs):
        """
        inputs: (batch, seq_len, d_model)
        Returns: (batch, seq_len, d_model)
        """
        x = inputs
        for layer in self.layers:
            x = layer(x)
        
        return self.norm(x)


class _BiMamba2(nn.Module):
    def __init__(self,
                 cin: int,
                 cout: int,
                 d_model: int,  # model dimension (D)
                 n_layer: int = 24,  # number of Mamba-2 layers in the language model
                 d_state: int = 128,  # state dimension (N)
                 d_conv: int = 4,  # convolution kernel size
                 expand: int = 2,  # expansion factor (E)
                 headdim: int = 64,  # head dimension (P)
                 chunk_size: int = 64,  # matrix partition size (Q)
                 ):
        super().__init__()
        self.fc_in = nn.Linear(cin, d_model, bias=False)  # 调整通道数到cmid
        self.mamba2_for = Mamba2(d_model, n_layer, d_state, d_conv, expand, headdim, chunk_size, )  # 正向
        self.mamba2_back = Mamba2(d_model, n_layer, d_state, d_conv, expand, headdim, chunk_size, )  # 负向
        self.fc_out = nn.Linear(d_model, cout, bias=False)  # 调整通道数到cout
        self.chunk_size = chunk_size

    @abstractmethod
    def forward(self, x):
        pass


class BiMamba2_1D(_BiMamba2):
    def __init__(self, cin, cout, d_model, **mamba2_args):
        super().__init__(cin, cout, d_model, **mamba2_args)

    def forward(self, x):
        l = x.shape[2]
        x = F.pad(x, (0, (64 - x.shape[2] % 64) % 64))  # 将 l , pad到4的倍数, [b, c64,l4]
        x = x.transpose(1, 2)  # 转成 1d 信号 [b, c64, d4*w4*h4]
        x = self.fc_in(x)  # 调整通道数为目标通道数
        x1 = self.mamba2_for(x)
        x2 = self.mamba2_back(x.flip(1)).flip(1)
        x = x1 + x2
        x = self.fc_out(x)  # 调整通道数为目标通道数
        x = x.transpose(1, 2)  # 转成 1d 信号 [b, c64, d4*w4*h4] ]
        x = x[:, :, :l]  # 截取原图大小
        return x


class BiMamba2_2D(_BiMamba2):
    def __init__(self, cin, cout, d_model, **mamba2_args):
        super().__init__(cin, cout, d_model, **mamba2_args)

    def forward(self, x):
        h, w = x.shape[2:]
        x = F.pad(x, (0, (8 - x.shape[3] % 8) % 8,
                      0, (8 - x.shape[2] % 8) % 8)
                  )  # 将 h , w  pad到8的倍数, [b, c64, h8, w8]
        _b, _c, _h, _w = x.shape
        x = x.permute(0, 2, 3, 1).reshape(_b, _h * _w, _c)
        x = self.fc_in(x)  # 调整通道数为目标通道数
        x1 = self.mamba2_for(x)
        x2 = self.mamba2_back(x.flip(1)).flip(1)
        x = x1 + x2
        x = self.fc_out(x)  # 调整通道数为目标通道数
        x = x.reshape(_b, _h, _w, -1, )
        x = x.permute(0, 3, 1, 2)
        x = x.reshape(_b, -1, _h, _w, )
        x = x[:, :, :h, :w]  # 截取原图大小
        return x


class BiMamba2_3D(_BiMamba2):
    def __init__(self, cin, cout, d_model, **mamba2_args):
        super().__init__(cin, cout, d_model, **mamba2_args)

    def forward(self, x):
        d, h, w = x.shape[2:]
        x = F.pad(x, (0, (4 - x.shape[4] % 4) % 4,
                      0, (4 - x.shape[3] % 4) % 4,
                      0, (4 - x.shape[2] % 4) % 4)
                  )  # 将 d, h, w , pad到4的倍数, [b, c64,d4, h4, w4]
        _b, _c, _d, _h, _w = x.shape
        x = x.permute(0, 2, 3, 4, 1).reshape(_b, _d * _h * _w, _c)
        x = self.fc_in(x)  # 调整通道数为目标通道数
        x1 = self.mamba2_for(x)
        x2 = self.mamba2_back(x.flip(1)).flip(1)
        x = x1 + x2
        x = self.fc_out(x)  # 调整通道数为目标通道数
        x = x.reshape(_b, _d, _h, _w, -1)
        x = x.permute(0, 4, 1, 2, 3)
        x=x.reshape(_b, -1, _d, _h, _w, )
        x = x[:, :, :d, :h, :w]  # 截取原图大小
        return x


class BiMamba2(_BiMamba2):
    def __init__(self, cin, cout, d_model, **mamba2_args):
        super().__init__(cin, cout, d_model, **mamba2_args)

    def forward(self, x):
        size = x.shape[2:]
        out_size = list(x.shape)
        out_size[1] = -1

        x = torch.flatten(x, 2)  # b c size
        l = x.shape[2]
        _s = self.chunk_size
        x = F.pad(x, [0, (_s - x.shape[2] % _s) % _s])  # 将 l, pad到chunk_size的倍数, [b, c64,l4]
        x = x.transpose(1, 2)  # 转成 1d 信号
        x = self.fc_in(x)  # 调整通道数为目标通道数
        x1 = self.mamba2_for(x)
        x2 = self.mamba2_back(x.flip(1)).flip(1)
        x = x1 + x2
        x = self.fc_out(x)  # 调整通道数为目标通道数
        x = x.transpose(1, 2)  # 转成 1d 信号
        x = x[:, :, :l]  # 截取原图大小
        x = x.reshape(out_size)

        return x


# For backward compatibility, keep SimpleBiMamba2 as an alternative
class SimpleBiMamba2(nn.Module):
    """Simplified bidirectional processing using LSTM for fallback"""
    def __init__(
        self,
        d_model: int,
        hidden_dim: int = None,
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




if __name__ == '__main__':
    # Test the new BiMambaBlocks2 with your input dimensions (7, 251, 64)
    print("Testing BiMambaBlocks2...")
    net_blocks = BiMambaBlocks2(encoder_dim=64, num_layers=2, d_state=64, chunk_size=32)
    x_blocks = torch.randn(7, 251, 64)
    y_blocks = net_blocks(x_blocks)
    print(f"BiMambaBlocks2 input shape: {x_blocks.shape}, output shape: {y_blocks.shape}")
    
    # Test BidirectionalMamba2_v2 single layer
    print("\nTesting BidirectionalMamba2_v2 single layer...")
    net_single = BidirectionalMamba2_v2(d_model=64, d_state=64, chunk_size=32)
    y_single = net_single(x_blocks)
    print(f"BidirectionalMamba2_v2 input shape: {x_blocks.shape}, output shape: {y_single.shape}")
    
    # Test original BiMamba2_1D (需要转换输入格式)
    print("\nTesting BiMamba2_1D...")
    # BiMamba2_1D expects (batch, channels, length), so we transpose
    x_1d = x_blocks.transpose(1, 2)  # (7, 64, 251)
    net_n = BiMamba2_1D(cin=64, cout=64, d_model=32)
    y_1d = net_n(x_1d)
    print(f"BiMamba2_1D input shape: {x_1d.shape}, output shape: {y_1d.shape}")
    
    # Test BiMamba2 general version
    print("\nTesting BiMamba2 general version...")
    net_general = BiMamba2(cin=64, cout=64, d_model=32, chunk_size=32)
    x_general = x_blocks.transpose(1, 2)  # (7, 64, 251)
    y_general = net_general(x_general)
    print(f"BiMamba2 general input shape: {x_general.shape}, output shape: {y_general.shape}")