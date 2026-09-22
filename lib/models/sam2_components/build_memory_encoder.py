import torch.nn as nn
from .memory_encoder import MemoryEncoder, MaskDownSampler, Fuser, CXBlock


def build_memory_encoder(
    out_dim=768,  # 输出维度改为768，用于后续交叉注意力
    in_dim=768,   # STTrack的opt_feat特征维度为768
    mask_downsampler_kernel_size=3,
    mask_downsampler_stride=2,
    mask_downsampler_padding=1,
    fuser_num_layers=2,
):
    """
   
    参数:
        out_dim: 输出特征维度 (默认: 768，用于与特征进行交叉注意力)
        in_dim: 输入图像特征维度 (默认: 768，对应STTrack的opt_feat)
        mask_downsampler_kernel_size: mask下采样卷积核大小 (默认: 3)
        mask_downsampler_stride: mask下采样步长 (默认: 2)
        mask_downsampler_padding: mask下采样padding (默认: 1)
        fuser_num_layers: Fuser层数 (默认: 2)
    
    返回:
        MemoryEncoder实例
    
    """
    
    # 创建mask下采样器
    # 从256x256下采样到16x16，需要下采样16倍
    # total_stride=16，使用stride=2，需要4层 (2^4=16)
    mask_downsampler = MaskDownSampler(
        embed_dim=in_dim,
        kernel_size=mask_downsampler_kernel_size,
        stride=mask_downsampler_stride,
        padding=mask_downsampler_padding,
        total_stride=16,  # 256 -> 16需要16倍下采样
        activation=nn.GELU,
    )
    
    # 创建Fuser
    fuser_layer = CXBlock(
        dim=in_dim,
        kernel_size=7,
        padding=3,
        layer_scale_init_value=1e-6,
        use_dwconv=True,
    )
    fuser = Fuser(
        layer=fuser_layer,
        num_layers=fuser_num_layers,
    )
    
    # 创建MemoryEncoder（不需要位置编码）
    memory_encoder = MemoryEncoder(
        out_dim=out_dim,
        mask_downsampler=mask_downsampler,
        fuser=fuser,
        in_dim=in_dim,
    )
    
    return memory_encoder
