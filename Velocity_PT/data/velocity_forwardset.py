"""
速度场预测的数据前处理模块
将数据字典转换为模型输入和标签
"""

import torch


def data_dict_to_input(data_dict):
    """
    将数据字典转换为模型输入
    
    输入:
        data_dict: 包含以下键的字典
            - sdf: SDF体素数据 [B, D, H, W]
            - sdf_query_points: 查询点坐标 [3, D, H, W]
            - vertices: 顶点坐标 [B, N, 3]
            - velocity: 速度场数据 [B, N, 3] (可选，训练时使用)
    
    输出:
        input_grid_features: 模型输入 [B, 4, D, H, W] (1通道SDF + 3通道坐标)
        output_points: 输出点坐标 [B, N, 3]
    """
    # SDF数据增加通道维度 [B, D, H, W] -> [B, 1, D, H, W]
    input_grid_features = data_dict['sdf'].unsqueeze(1)
    
    # 查询点坐标 [3, D, H, W]
    grid_points = data_dict['sdf_query_points']
    
    # 拼接SDF和坐标 [B, 1, D, H, W] + [3, D, H, W] -> [B, 4, D, H, W]
    # 需要扩展grid_points以匹配batch维度
    if grid_points.dim() == 4:
        grid_points = grid_points.unsqueeze(0).expand(input_grid_features.shape[0], -1, -1, -1, -1)
    
    input_grid_features = torch.cat(
        tensors=(input_grid_features, grid_points),
        dim=1
    )
    
    output_points = data_dict['vertices']  # [B, N, 3]
    
    return input_grid_features, output_points


def loss_dict(data_dict):
    """
    将数据字典转换为损失函数所需的输入和目标
    
    输入:
        data_dict: 包含以下键的字典
            - sdf: SDF体素数据
            - sdf_query_points: 查询点坐标
            - vertices: 顶点坐标 [B, N, 3]
            - velocity: 速度场数据 [B, N, 3]
    
    输出:
        input_grid_features: 模型输入 [B, 4, D, H, W]
        output_points: 顶点坐标 [B, N, 3]
        true_velocity: 真实速度场 [B, N, 3] 或 None
    """
    input_grid_features, output_points = data_dict_to_input(data_dict)
    
    true_velocity = None
    if 'velocity' in data_dict.keys():
        true_velocity = data_dict['velocity']  # [B, N, 3]
    
    return input_grid_features, output_points, true_velocity


def velocity_to_components(velocity_flat, num_vertices):
    """
    将展平的速度向量拆分为三个分量
    
    输入:
        velocity_flat: [B, 3*N] 展平的速度向量
        num_vertices: 顶点数量 N
    
    输出:
        vx, vy, vz: 各 [B, N] 三个速度分量
    """
    B = velocity_flat.shape[0]
    velocity_3d = velocity_flat.reshape(B, num_vertices, 3)
    vx = velocity_3d[:, :, 0]
    vy = velocity_3d[:, :, 1]
    vz = velocity_3d[:, :, 2]
    return vx, vy, vz


def components_to_velocity(vx, vy, vz):
    """
    将三个速度分量合并为展平的速度向量
    
    输入:
        vx, vy, vz: 各 [B, N] 三个速度分量
    
    输出:
        velocity_flat: [B, 3*N] 展平的速度向量
    """
    B, N = vx.shape
    velocity_3d = torch.stack([vx, vy, vz], dim=-1)  # [B, N, 3]
    velocity_flat = velocity_3d.reshape(B, -1)  # [B, 3*N]
    return velocity_flat


if __name__ == "__main__":
    # 测试代码
    batch_size = 2
    num_vertices = 100
    spatial_res = 64
    
    # 模拟数据字典
    data_dict = {
        'sdf': torch.randn(batch_size, spatial_res, spatial_res, spatial_res),
        'sdf_query_points': torch.randn(3, spatial_res, spatial_res, spatial_res),
        'vertices': torch.randn(batch_size, num_vertices, 3),
        'velocity': torch.randn(batch_size, num_vertices * 3)
    }
    
    # 测试loss_dict
    input_features, true_velocity = loss_dict(data_dict)
    print(f"输入特征形状: {input_features.shape}")  # 应为 [2, 4, 64, 64, 64]
    print(f"真实速度形状: {true_velocity.shape}")   # 应为 [2, 300]
    
    # 测试速度分量拆分和合并
    vx, vy, vz = velocity_to_components(true_velocity, num_vertices)
    print(f"速度分量形状: vx={vx.shape}, vy={vy.shape}, vz={vz.shape}")
    
    velocity_reconstructed = components_to_velocity(vx, vy, vz)
    print(f"重建速度形状: {velocity_reconstructed.shape}")
    print(f"重建误差: {torch.max(torch.abs(velocity_reconstructed - true_velocity)).item()}")
