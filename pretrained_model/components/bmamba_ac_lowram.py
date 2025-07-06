import torch
from torch import nn
import torch
from torch import Tensor, nn
from torch.nn import functional as F
from torch.utils.checkpoint import checkpoint
from abc import abstractmethod


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
        
        # 添加参数初始化
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
        # 使用梯度检查点来节省显存
        if self.training:
            return checkpoint(self._forward_impl, u, use_reentrant=False)
        else:
            return self._forward_impl(u)
    
    def _forward_impl(self, u: Tensor):
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
        """
        更高效的segsum实现，避免创建大的T×T矩阵
        """
        T = x.size(-1)
        device = x.device
        
        # 如果T很大，使用分块处理
        if T > 128:
            return self._segsum_chunked(x, chunk_size=64)
        
        # 原始实现（仅用于小序列）
        x = x[..., None].repeat(1, 1, 1, 1, T)
        mask = torch.tril(torch.ones(T, T, dtype=torch.bool, device=device), diagonal=-1)
        x = x.masked_fill(~mask, 0)
        x_segsum = torch.cumsum(x, dim=-2)
        mask = torch.tril(torch.ones(T, T, dtype=torch.bool, device=device), diagonal=0)
        x_segsum = x_segsum.masked_fill(~mask, -1e9)
        return x_segsum
    
    def _segsum_chunked(self, x: Tensor, chunk_size: int = 64) -> Tensor:
        """
        分块计算segsum，大幅减少显存占用
        """
        T = x.size(-1)
        device = x.device
        batch_shape = x.shape[:-1]
        
        # 初始化输出
        result = torch.zeros(*batch_shape, T, T, device=device, dtype=x.dtype)
        
        # 分块处理
        for i in range(0, T, chunk_size):
            end_i = min(i + chunk_size, T)
            for j in range(0, T, chunk_size):
                end_j = min(j + chunk_size, T)
                
                # 只处理下三角部分
                if i >= j:
                    chunk_x = x[..., i:end_i, None].repeat(1, 1, 1, 1, end_j - j)
                    chunk_mask = torch.tril(
                        torch.ones(end_i - i, end_j - j, dtype=torch.bool, device=device),
                        diagonal=j - i - 1
                    )
                    chunk_x = chunk_x.masked_fill(~chunk_mask, 0)
                    result[..., i:end_i, j:end_j] = torch.cumsum(chunk_x, dim=-2)
        
        # 应用最终mask
        final_mask = torch.tril(torch.ones(T, T, dtype=torch.bool, device=device), diagonal=0)
        result = result.masked_fill(~final_mask, -1e9)
        
        return result

    def ssd(self, x, A, B, C):
        chunk_size = self.chunk_size
        x = x.reshape(x.shape[0], x.shape[1] // chunk_size, chunk_size, x.shape[2], x.shape[3], )
        B = B.reshape(B.shape[0], B.shape[1] // chunk_size, chunk_size, B.shape[2], B.shape[3], )
        C = C.reshape(C.shape[0], C.shape[1] // chunk_size, chunk_size, C.shape[2], C.shape[3], )
        A = A.reshape(A.shape[0], A.shape[1] // chunk_size, chunk_size, A.shape[2])
        A = A.permute(0, 3, 1, 2)
        A_cumsum = torch.cumsum(A, dim=-1)

        # 1. Compute the output for each intra-chunk (diagonal blocks)
        # 添加数值稳定性
        L = torch.exp(torch.clamp(self.segsum(A), min=-10, max=10))
        Y_diag = torch.einsum("bclhn, bcshn, bhcls, bcshp -> bclhp", C, B, L, x)

        # 2. Compute the state for each intra-chunk
        decay_states = torch.exp(torch.clamp(A_cumsum[:, :, :, -1:] - A_cumsum, min=-10, max=10))
        states = torch.einsum("bclhn, bhcl, bclhp -> bchpn", B, decay_states, x)

        # 3. Compute the inter-chunk SSM recurrence
        initial_states = torch.zeros_like(states[:, :1])
        states = torch.cat([initial_states, states], dim=1)

        decay_chunk = torch.exp(torch.clamp(self.segsum(F.pad(A_cumsum[:, :, :, -1], (1, 0))), min=-10, max=10))[0]
        new_states = torch.einsum("bhzc, bchpn -> bzhpn", decay_chunk, states)
        states = new_states[:, :-1]

        # 4. Compute state -> output conversion per chunk
        state_decay_out = torch.exp(torch.clamp(A_cumsum, min=-10, max=10))
        Y_off = torch.einsum("bclhn, bchpn, bhcl -> bclhp", C, states, state_decay_out)

        # Add output of intra-chunk and inter-chunk terms (diagonal and off-diagonal blocks)
        Y = Y_diag + Y_off
        Y = Y.reshape(Y.shape[0], Y.shape[1] * Y.shape[2], Y.shape[3], Y.shape[4], )

        return Y


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
        x = F.pad(x, (0, (64 - x.shape[2] % 64) % 64))  # 将 l , pad到64的倍数
        x = x.transpose(1, 2)  # 转成 1d 信号 [b, l, c]
        x = self.fc_in(x)  # 调整通道数为目标通道数
        x1 = self.mamba2_for(x)
        x2 = self.mamba2_back(x.flip(1)).flip(1)
        x = x1 + x2
        x = self.fc_out(x)  # 调整通道数为目标通道数
        x = x.transpose(1, 2)  # 转回 [b, c, l]
        x = x[:, :, :l]  # 截取原图大小
        return x


# 双向非对称mamba2 块
class BiMamba2Ac2d(nn.Module):
    def __init__(self, cin, cout, d_model, **mamba2_args):
        super().__init__()
        self.bi_mamba_h1 = BiMamba2_1D(cout, cout, d_model, **mamba2_args)
        self.bi_mamba_w1 = BiMamba2_1D(cin, cout, d_model, **mamba2_args)

        self.bi_mamba_h2 = BiMamba2_1D(cin, cout, d_model, **mamba2_args)
        self.bi_mamba_w2 = BiMamba2_1D(cout, cout, d_model, **mamba2_args)

        self.fc = nn.Conv2d(cout * 2, cout, 1)

    def forward(self, x):
        # 在训练时使用梯度检查点
        if self.training and x.requires_grad:
            return checkpoint(self._forward_impl, x, use_reentrant=False)
        else:
            return self._forward_impl(x)
    
    def _forward_impl(self, x):
        # 非对称卷积
        b, c, h, w = x.shape

        # 先 w 再 h
        x_w1 = x.transpose(1, 2).reshape(b * h, c, w)  # bh, c, w
        y_w1 = self.bi_mamba_w1(x_w1)
        y_w1 = y_w1.reshape(b, h, -1, w).transpose(1, 2)
        x_h1 = y_w1.transpose(1, 3).reshape(b * w, -1, h)  # bw, c, h
        y1 = self.bi_mamba_h1(x_h1).reshape(b, w, -1, h).transpose(1, 2).transpose(2, 3)

        # 先 h 再 w
        x_h2 = x.transpose(1, 3).reshape(b * w, c, h)  # bw, c, h
        y_h2 = self.bi_mamba_h2(x_h2).reshape(b, w, -1, h).transpose(1, 2).transpose(2, 3)
        x_w2 = y_h2.transpose(1, 2).reshape(b * h, -1, w)  # bh, c, w
        y2 = self.bi_mamba_w2(x_w2).reshape(b, h, -1, w).transpose(1, 2)

        # 合并结果
        y = torch.cat([y1, y2], dim=1)
        y = self.fc(y)
        return y


# ===== 适配器类：用于CNN14系统 =====

class BMamba2DAcBlocks(nn.Module):
    """
    2D非对称BiMamba blocks，适配CNN14 encoder作为decoder使用
    兼容ConformerBlocks接口
    """
    def __init__(
        self,
        encoder_dim: int = 512,
        num_layers: int = 2,
        d_state: int = 64,
        d_conv: int = 8,
        expand: int = 2,
        headdim: int = 32,
        chunk_size: int = 64,
        dropout: float = 0.1,
        d_model: int = 32,
        # 2D特定参数
        time_dim: int = 251,    # 时间维度
        freq_dim: int = 64,     # 频率维度（假设输入的最后一维是频率特征）
        **kwargs
    ):
        super().__init__()
        
        self.encoder_dim = encoder_dim
        self.time_dim = time_dim
        self.freq_dim = freq_dim
        self.d_model = d_model
        
        # 确保headdim兼容性
        d_inner = expand * d_model
        if d_inner % headdim != 0:
            headdim = 32  # 安全值
            while d_inner % headdim != 0 and headdim > 1:
                headdim //= 2
            if headdim < 1:
                headdim = 1
            print(f"Warning: Adjusted headdim to {headdim} for compatibility")
        
        # 输入适配：从1D序列转换为2D
        self.input_adapter = nn.Sequential(
            nn.Linear(encoder_dim, freq_dim),  # 调整特征维度
            nn.LayerNorm(freq_dim),
            nn.Dropout(dropout)
        )
        
        # 2D Mamba层
        self.mamba_layers = nn.ModuleList([
            BiMamba2Ac2d(
                cin=1 if i == 0 else freq_dim,  # 第一层输入1通道，后续层输入freq_dim通道
                cout=freq_dim,
                d_model=d_model,
                n_layer=1,  # 每个BiMamba2Ac2d内部只有1层
                d_state=d_state,
                d_conv=d_conv,
                expand=expand,
                headdim=headdim,
                chunk_size=chunk_size,
            )
            for i in range(num_layers)
        ])
        
        # 输出适配：从2D转换回1D序列
        self.output_adapter = nn.Sequential(
            nn.Conv2d(freq_dim, 1, 1),  # 降维到1通道
            nn.AdaptiveAvgPool2d((time_dim, encoder_dim)),  # 调整到目标尺寸
        )
        
        # 最终输出层
        self.final_norm = nn.LayerNorm(encoder_dim)
        self.dropout = nn.Dropout(dropout)
        
        print(f"BMamba2DAcBlocks initialized:")
        print(f"  encoder_dim: {encoder_dim}, num_layers: {num_layers}")
        print(f"  time_dim: {time_dim}, freq_dim: {freq_dim}")
        print(f"  d_state: {d_state}, d_conv: {d_conv}, d_model: {d_model}")
        print(f"  headdim: {headdim}, chunk_size: {chunk_size}")

    def forward(self, inputs):
        """
        输入: (batch, seq_len, encoder_dim) 例如 (7, 251, 64)
        输出: (batch, seq_len, encoder_dim)
        """
        # 在训练时使用梯度检查点
        if self.training and inputs.requires_grad:
            return checkpoint(self._forward_impl, inputs, use_reentrant=False)
        else:
            return self._forward_impl(inputs)
    
    def _forward_impl(self, inputs):
        batch_size, seq_len, _ = inputs.shape
        
        # 1. 输入适配：1D → 2D
        x = self.input_adapter(inputs)  # (batch, seq_len, freq_dim)
        
        # 重塑为2D格式: (batch, 1, seq_len, freq_dim)
        x = x.unsqueeze(1)  # (batch, 1, time_dim, freq_dim)
        
        # 2. 2D Mamba处理（使用更保守的残差连接）
        for i, layer in enumerate(self.mamba_layers):
            if i == 0:
                # 第一层不使用残差连接（维度可能不匹配）
                x = layer(x)
            else:
                # 后续层使用残差连接
                residual = x
                x = layer(x)
                if x.shape == residual.shape:
                    x = x + residual
        
        # 3. 输出适配：2D → 1D
        x = self.output_adapter(x)  # (batch, 1, time_dim, encoder_dim)
        x = x.squeeze(1)  # (batch, time_dim, encoder_dim)
        
        # 确保输出维度正确
        if x.shape[1] != seq_len:
            x = F.interpolate(
                x.transpose(1, 2), 
                size=seq_len, 
                mode='linear', 
                align_corners=False
            ).transpose(1, 2)
        
        # 4. 最终处理
        x = self.final_norm(x)
        x = self.dropout(x)
        
        # 残差连接
        x = x + inputs
        
        return x


# 简化的测试功能（替代torchnssd）
def simple_test():
    """简单的测试函数"""
    print("Testing BMamba2DAcBlocks...")
    
    # 创建模型
    model = BMamba2DAcBlocks(
        encoder_dim=64,
        num_layers=2,
        d_state=32,
        d_conv=8,
        d_model=16,
        time_dim=251,
        freq_dim=64
    )
    
    # 测试输入 (符合你的SELD场景)
    x = torch.randn(7, 251, 64)
    print(f"Input shape: {x.shape}")
    
    # 前向传播
    model.eval()
    with torch.no_grad():
        output = model(x)
    
    print(f"Output shape: {output.shape}")
    print(f"Model parameters: {sum(p.numel() for p in model.parameters()):,}")
    
    return model, output

def test_memory_usage():
    """测试显存使用情况"""
    if not torch.cuda.is_available():
        print("CUDA not available, skipping memory test")
        return
        
    print("Testing memory usage...")
    device = torch.device('cuda')
    
    # 清空缓存
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats(device)
    
    # 创建模型
    model = BMamba2DAcBlocks(
        encoder_dim=512,  # 更大的模型
        num_layers=2,
        d_state=64,
        d_conv=8,
        d_model=32,
        time_dim=251,
        freq_dim=64
    ).to(device)
    
    # 模拟训练时的batch_size
    batch_size = 32
    x = torch.randn(batch_size, 251, 512, device=device, requires_grad=True)
    
    print(f"Initial memory: {torch.cuda.memory_allocated(device) / 1e9:.2f} GB")
    
    # 前向传播
    model.train()
    output = model(x)
    print(f"After forward: {torch.cuda.memory_allocated(device) / 1e9:.2f} GB")
    
    # 反向传播
    loss = output.sum()
    loss.backward()
    print(f"After backward: {torch.cuda.memory_allocated(device) / 1e9:.2f} GB")
    print(f"Peak memory: {torch.cuda.max_memory_allocated(device) / 1e9:.2f} GB")
    
    # 清理
    del model, x, output, loss
    torch.cuda.empty_cache()
    
    print("Memory test completed.")


if __name__ == '__main__':
    # 运行简单测试
    model, output = simple_test()
    
    # 运行显存测试
    test_memory_usage()