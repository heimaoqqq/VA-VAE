"""
最小化VA-VAE修改 - 仅添加用户条件功能
基于LightningDiT/tokenizer/vavae.py的最小修改
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class UserConditionedVAVAE(nn.Module):
    """
    用户条件化的VA-VAE
    在原有VA-VAE基础上添加最简单的用户条件功能
    """
    
    def __init__(self, original_vavae, num_users=31, condition_dim=128):
        """
        基于原有VA-VAE添加用户条件
        
        Args:
            original_vavae: 原始的VA-VAE模型
            num_users: 用户数量
            condition_dim: 条件向量维度
        """
        super().__init__()
        
        # 保持原有的VA-VAE结构
        self.encoder = original_vavae.encoder
        self.decoder = original_vavae.decoder
        self.quant_conv = original_vavae.quant_conv
        self.post_quant_conv = original_vavae.post_quant_conv
        
        # 仅添加用户嵌入层
        self.num_users = num_users
        self.condition_dim = condition_dim
        self.user_embedding = nn.Embedding(num_users, condition_dim)
        
        # 简单的条件融合层
        # 获取编码器输出的通道数
        encoder_out_channels = self._get_encoder_out_channels()
        self.condition_proj = nn.Linear(condition_dim, encoder_out_channels)
        
        print(f"添加用户条件: {num_users}个用户, 条件维度: {condition_dim}")
    
    def _get_encoder_out_channels(self):
        """获取编码器输出通道数"""
        # 创建一个dummy输入来获取编码器输出形状
        dummy_input = torch.randn(1, 3, 256, 256)
        with torch.no_grad():
            encoder_out = self.encoder(dummy_input)
            return encoder_out.shape[1]
    
    def encode(self, x, user_ids=None):
        """
        编码 - 添加用户条件
        
        Args:
            x: 输入图像 (B, 3, H, W)
            user_ids: 用户ID (B,)
            
        Returns:
            posterior: 后验分布
        """
        # 原有的编码过程
        h = self.encoder(x)
        
        # 添加用户条件 (如果提供)
        if user_ids is not None:
            # 获取用户嵌入
            user_emb = self.user_embedding(user_ids)  # (B, condition_dim)
            
            # 投影到编码器输出空间
            user_cond = self.condition_proj(user_emb)  # (B, encoder_channels)
            
            # 简单的特征融合: 逐元素相加
            # 扩展维度以匹配空间维度
            user_cond = user_cond.unsqueeze(-1).unsqueeze(-1)  # (B, C, 1, 1)
            h = h + user_cond  # 广播相加
        
        # 原有的量化卷积
        moments = self.quant_conv(h)
        
        # 返回后验分布 (保持原有格式)
        return DiagonalGaussianDistribution(moments)
    
    def decode(self, z, user_ids=None):
        """
        解码 - 添加用户条件
        
        Args:
            z: 潜在变量
            user_ids: 用户ID
            
        Returns:
            重建图像
        """
        # 原有的后量化卷积
        z = self.post_quant_conv(z)
        
        # 添加用户条件 (如果提供)
        if user_ids is not None:
            # 获取用户嵌入
            user_emb = self.user_embedding(user_ids)
            
            # 投影到潜在空间
            user_cond = self.condition_proj(user_emb)
            user_cond = user_cond.unsqueeze(-1).unsqueeze(-1)
            z = z + user_cond
        
        # 原有的解码过程
        dec = self.decoder(z)
        return dec
    
    def forward(self, input, user_ids=None, sample_posterior=True):
        """
        前向传播 - 保持原有接口，添加用户条件
        
        Args:
            input: 输入图像
            user_ids: 用户ID
            sample_posterior: 是否从后验采样
            
        Returns:
            重建图像和其他信息
        """
        posterior = self.encode(input, user_ids)
        
        if sample_posterior:
            z = posterior.sample()
        else:
            z = posterior.mode()
        
        dec = self.decode(z, user_ids)
        
        return dec, posterior
    
    def get_last_layer(self):
        """获取最后一层 - 保持原有接口"""
        return self.decoder.conv_out.weight
    
    def sample(self, user_ids, num_samples=1, device='cuda'):
        """
        从先验分布采样生成
        
        Args:
            user_ids: 用户ID tensor (B,)
            num_samples: 每个用户生成的样本数
            device: 设备
            
        Returns:
            生成的图像
        """
        batch_size = user_ids.size(0)
        
        # 从标准正态分布采样
        # 需要知道潜在空间的形状，这里假设是 (B, C, H, W)
        # 可以通过编码一个样本来获取形状
        with torch.no_grad():
            dummy_input = torch.randn(1, 3, 256, 256, device=device)
            dummy_posterior = self.encode(dummy_input)
            latent_shape = dummy_posterior.sample().shape[1:]  # (C, H, W)
        
        # 为每个用户生成样本
        all_samples = []
        for i in range(num_samples):
            z = torch.randn(batch_size, *latent_shape, device=device)
            samples = self.decode(z, user_ids)
            all_samples.append(samples)
        
        return torch.cat(all_samples, dim=0)


# 用于兼容原有代码的后验分布类
class DiagonalGaussianDistribution:
    """对角高斯分布 - 保持与原有代码兼容"""
    
    def __init__(self, parameters):
        self.parameters = parameters
        self.mean, self.logvar = torch.chunk(parameters, 2, dim=1)
        self.logvar = torch.clamp(self.logvar, -30.0, 20.0)
        self.std = torch.exp(0.5 * self.logvar)
        self.var = torch.exp(self.logvar)
    
    def sample(self):
        x = self.mean + self.std * torch.randn_like(self.mean)
        return x
    
    def mode(self):
        return self.mean
    
    def kl(self, other=None):
        if other is None:
            return 0.5 * torch.sum(torch.pow(self.mean, 2) + self.var - 1.0 - self.logvar, dim=[1, 2, 3])
        else:
            return 0.5 * torch.sum(
                torch.pow(self.mean - other.mean, 2) / other.var + 
                self.var / other.var - 1.0 - self.logvar + other.logvar,
                dim=[1, 2, 3]
            )


def create_user_conditioned_vavae(original_vavae_path, num_users=31, condition_dim=128):
    """
    创建用户条件化的VA-VAE
    
    Args:
        original_vavae_path: 原始VA-VAE模型路径
        num_users: 用户数量
        condition_dim: 条件维度
        
    Returns:
        用户条件化的VA-VAE模型
    """
    # 加载原始VA-VAE
    print(f"加载原始VA-VAE模型: {original_vavae_path}")
    checkpoint = torch.load(original_vavae_path, map_location='cpu')
    
    # 这里需要根据实际的LightningDiT模型结构来调整
    # 假设原始模型在checkpoint['model']中
    original_vavae = checkpoint['model']  # 或者其他键名
    
    # 创建条件化模型
    conditioned_model = UserConditionedVAVAE(
        original_vavae=original_vavae,
        num_users=num_users,
        condition_dim=condition_dim
    )
    
    print(f"成功创建用户条件化VA-VAE，支持{num_users}个用户")
    
    return conditioned_model


# 使用示例
if __name__ == "__main__":
    # 测试用户条件化VA-VAE
    print("测试用户条件化VA-VAE...")
    
    # 创建一个简单的测试
    # 注意：这里需要实际的VA-VAE模型来测试
    
    # 模拟输入
    batch_size = 4
    num_users = 31
    
    # 模拟图像和用户ID
    images = torch.randn(batch_size, 3, 256, 256)
    user_ids = torch.randint(0, num_users, (batch_size,))
    
    print(f"输入图像形状: {images.shape}")
    print(f"用户ID: {user_ids}")
    
    # 这里需要实际的模型来测试
    # conditioned_model = create_user_conditioned_vavae("path/to/vavae.pth")
    # output, posterior = conditioned_model(images, user_ids)
    # print(f"输出形状: {output.shape}")
    
    print("测试完成！")
