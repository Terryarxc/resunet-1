"""
速度场预测专用 ResNet-UNet 模型
基于原压力场模型架构，使用 grid_sample + MLP 输出，支持任意数量顶点
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


# CBAM注意力机制模块
class ChannelAttention(nn.Module):
    def __init__(self, in_channels, reduction_ratio=16):
        super(ChannelAttention, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool3d(1)
        self.max_pool = nn.AdaptiveMaxPool3d(1)
        
        self.fc = nn.Sequential(
            nn.Conv3d(in_channels, in_channels // reduction_ratio, 1, bias=False),
            nn.ReLU(inplace=False),
            nn.Conv3d(in_channels // reduction_ratio, in_channels, 1, bias=False)
        )
        self.sigmoid = nn.Sigmoid()
        
    def forward(self, x):
        avg_out = self.fc(self.avg_pool(x))
        max_out = self.fc(self.max_pool(x))
        out = avg_out + max_out
        return self.sigmoid(out)


class SpatialAttention(nn.Module):
    def __init__(self, kernel_size=7):
        super(SpatialAttention, self).__init__()
        padding = kernel_size // 2
        self.conv = nn.Conv3d(2, 1, kernel_size, padding=padding, bias=False)
        self.sigmoid = nn.Sigmoid()
        
    def forward(self, x):
        avg_out = torch.mean(x, dim=1, keepdim=True)
        max_out, _ = torch.max(x, dim=1, keepdim=True)
        x_concat = torch.cat([avg_out, max_out], dim=1)
        x_conv = self.conv(x_concat)
        return self.sigmoid(x_conv)


class CBAM(nn.Module):
    def __init__(self, in_channels, reduction_ratio=16, kernel_size=7):
        super(CBAM, self).__init__()
        self.channel_att = ChannelAttention(in_channels, reduction_ratio)
        self.spatial_att = SpatialAttention(kernel_size)
        
    def forward(self, x):
        x_ca = x * self.channel_att(x)
        x_out = x_ca * self.spatial_att(x_ca)
        return x_out


class BasicBlock(nn.Module):
    """带CBAM注意力的残差块"""
    def __init__(self, in_channels, out_channels, stride=1):
        super(BasicBlock, self).__init__()
        self.conv1 = nn.Conv3d(in_channels, out_channels, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm3d(out_channels)
        self.relu = nn.ReLU(inplace=False)
        self.conv2 = nn.Conv3d(out_channels, out_channels, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm3d(out_channels)
        
        self.cbam = CBAM(out_channels, reduction_ratio=8, kernel_size=5)
        
        self.shortcut = nn.Identity()
        if stride != 1 or in_channels != out_channels:
            self.shortcut = nn.Sequential(
                nn.Conv3d(in_channels, out_channels, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm3d(out_channels)
            )
    
    def forward(self, x):
        identity = x
        
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        
        out = self.conv2(out)
        out = self.bn2(out)
        out = self.cbam(out)
        
        out = out + self.shortcut(identity)
        out = self.relu(out)
        
        return out


class DecoderBlock(nn.Module):
    """U-Net解码器块"""
    def __init__(self, in_channels, skip_channels, out_channels, dropout_rate=0.24):
        super(DecoderBlock, self).__init__()
        self.upsample = nn.Upsample(scale_factor=2, mode='trilinear', align_corners=True)
        self.conv1 = nn.Conv3d(in_channels + skip_channels, out_channels, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm3d(out_channels)
        self.relu = nn.ReLU(inplace=False)
        self.conv2 = nn.Conv3d(out_channels, out_channels, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm3d(out_channels)
        self.dropout = nn.Dropout3d(dropout_rate)
        self.cbam = CBAM(out_channels, reduction_ratio=8, kernel_size=5)

    def forward(self, x, skip=None):
        x_up = self.upsample(x)
        if skip is not None:
            if x_up.shape[2:] != skip.shape[2:]:
                x_up = F.interpolate(x_up, size=skip.shape[2:], mode='trilinear', align_corners=True)
            x_cat = torch.cat([x_up, skip], dim=1)
            x_conv = self.conv1(x_cat)
        else:
            x_conv = self.conv1(x_up)
            
        x_bn = self.bn1(x_conv)
        x_relu = self.relu(x_bn)
        x_drop = self.dropout(x_relu)
        x_conv2 = self.conv2(x_drop)
        x_bn2 = self.bn2(x_conv2)
        x_cbam = self.cbam(x_bn2)
        x_out = self.relu(x_cbam)
        return x_out


class ResNetUNetVelocity(nn.Module):
    """
    速度场预测专用模型
    
    与压力场模型的区别：
    - 使用 grid_sample 在任意顶点位置采样特征
    - 通过 MLP 输出每个顶点的速度向量 (vx, vy, vz)
    - 支持任意数量的顶点
    """
    def __init__(self, in_channels=4, out_channels=3, hidden_channels=48, use_attention=True):
        super(ResNetUNetVelocity, self).__init__()
        self.use_attention = use_attention
        self.hidden_channels = hidden_channels
        self.out_channels = out_channels
        
        # 编码器部分
        self.stem = nn.Sequential(
            nn.Conv3d(in_channels, 48, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm3d(48),
            nn.ReLU(inplace=False)
        )
        
        self.layer1 = self._make_layer(BasicBlock, 48, 48, blocks=2, stride=1)
        self.layer2 = self._make_layer(BasicBlock, 48, 64, blocks=2, stride=2)
        self.layer3 = self._make_layer(BasicBlock, 64, 128, blocks=2, stride=2)
        self.layer4 = self._make_layer(BasicBlock, 128, 256, blocks=2, stride=2)
        
        # 解码器块
        self.decoder4 = DecoderBlock(256, 128, 128, dropout_rate=0.24)
        self.decoder3 = DecoderBlock(128, 64, 64, dropout_rate=0.24)
        self.decoder2 = DecoderBlock(64, 48, 48, dropout_rate=0.24)
        
        # 最终输出层 - 输出hidden_channels通道的特征图
        self.final_conv = nn.Conv3d(48, hidden_channels, kernel_size=3, padding=1)
        self.final_bn = nn.BatchNorm3d(hidden_channels)
        self.final_relu = nn.ReLU(inplace=False)
        
        # MLP输出头 - 将采样的特征映射到速度向量
        self.output_mlp = nn.Sequential(
            nn.Linear(hidden_channels, hidden_channels),
            nn.GELU(),
            nn.Linear(hidden_channels, out_channels)
        )
    
    def _make_layer(self, block, in_channels, out_channels, blocks, stride=1):
        layers = []
        layers.append(block(in_channels, out_channels, stride))
        for _ in range(1, blocks):
            layers.append(block(out_channels, out_channels, 1))
        return nn.Sequential(*layers)
    
    def forward(self, x, output_points):
        """
        前向传播
        
        Args:
            x: 输入特征 [B, 4, D, H, W] (SDF + 坐标)
            output_points: 顶点坐标 [B, N, 3]，已归一化到 [-1, 1]
        
        Returns:
            velocity: 预测的速度场 [B, N, 3]
        """
        # 存储跳跃连接特征
        skip_connections = []
        
        # 编码器
        x = self.stem(x)
        skip_connections.append(x)
        
        x = self.layer1(x)
        skip_connections.append(x)
        
        x = self.layer2(x)
        skip_connections.append(x)
        
        x = self.layer3(x)
        skip_connections.append(x)
        
        x = self.layer4(x)
        
        # 解码器
        x = self.decoder4(x, skip_connections[3])
        x = self.decoder3(x, skip_connections[2])
        x = self.decoder2(x, skip_connections[1])
        
        # 最终卷积层
        x = self.final_conv(x)
        x = self.final_bn(x)
        x = self.final_relu(x)
        # x: [B, hidden_channels, D', H', W']
        
        # 使用 grid_sample 在顶点位置采样特征
        # output_points: [B, N, 3] -> [B, N, 1, 1, 3] for grid_sample
        grid = output_points.unsqueeze(2).unsqueeze(2)  # [B, N, 1, 1, 3]
        
        # grid_sample 要求 grid 的最后一维是 (x, y, z) 对应 (W, H, D)
        # 输入坐标已经是 [-1, 1] 范围
        sampled_features = F.grid_sample(
            x, grid, 
            mode='bilinear', 
            padding_mode='border',
            align_corners=False
        )  # [B, hidden_channels, N, 1, 1]
        
        # 调整形状
        sampled_features = sampled_features.squeeze(-1).squeeze(-1)  # [B, hidden_channels, N]
        sampled_features = sampled_features.permute(0, 2, 1)  # [B, N, hidden_channels]
        
        # 通过MLP得到速度
        velocity = self.output_mlp(sampled_features)  # [B, N, 3]
        
        return velocity


if __name__ == "__main__":
    # 测试模型
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    model = ResNetUNetVelocity(in_channels=4, out_channels=3, hidden_channels=48).to(device)
    
    # 模拟输入
    batch_size = 1
    x = torch.randn(batch_size, 4, 64, 64, 64).to(device)  # SDF + 坐标
    output_points = torch.randn(batch_size, 29498, 3).to(device)  # 顶点坐标，归一化到[-1,1]
    output_points = torch.tanh(output_points)  # 确保在[-1, 1]范围
    
    # 前向传播
    with torch.no_grad():
        velocity = model(x, output_points)
    
    print(f"输入形状: {x.shape}")
    print(f"顶点数量: {output_points.shape[1]}")
    print(f"输出速度形状: {velocity.shape}")  # 应为 [1, 29498, 3]
    
    # 计算参数量
    total_params = sum(p.numel() for p in model.parameters())
    print(f"模型参数量: {total_params:,}")
