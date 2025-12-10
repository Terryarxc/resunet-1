# 速度场预测模块 (Velocity_PT)

基于改进的 ResNet-UNet 模型的三维速度场预测模块。

## 目录结构

```
Velocity_PT/
├── configs/
│   └── velocity_resnet.yaml        # 训练配置文件
├── data/
│   ├── velocity_datamodule_pt.py   # 数据模块（读取VTK速度数据）
│   └── velocity_forwardset.py      # 数据前处理
├── models/
│   └── resnet_unet_velocity.py     # 速度场专用模型
├── train_velocity_resnet.py        # 训练脚本
└── README.md
```

## 数据格式

- **训练/测试数据目录**: `data/train_velocity_vtk/`
- **索引文件目录**: `data/train_velocity/`
- **VTK文件命名**: `vel_###.vtk` 或 `mesh_###.vtk`
- **速度数据**: VTK文件中的 `point_vectors` 字段（3个分量：vx, vy, vz）
- **顶点数**: 29498个点/样本

## 使用方法

### 1. 训练模型

```bash
cd Velocity_PT
python train_velocity_resnet.py
```

### 2. 修改配置

编辑 `configs/velocity_resnet.yaml` 调整：
- `n_train`: 训练样本数量 (默认50)
- `n_test`: 测试样本数量 (默认10)
- `num_epochs`: 训练轮数 (默认100)
- `batch_size`: 批量大小 (默认1)
- `lr`: 学习率 (默认0.001)
- `hidden_channels`: 特征通道数 (默认48)

### 3. 输出

- 模型权重: `output/best_model-ResNet_UNet_Velocity.pth`
- 训练历史: `output/training_history_ResNet_UNet_Velocity.json`
- 预测结果: `output/Velocity_Dataset/vel_pred_###.npy`

## 模型架构

**ResNetUNetVelocity** - 速度场专用模型

与压力场模型的区别：
- 使用 `grid_sample` 在任意顶点位置采样特征
- 通过 MLP 输出每个顶点的速度向量 (vx, vy, vz)
- 支持任意数量的顶点（不需要固定输出维度）

架构流程：
1. 输入: [B, 4, 64, 64, 64] (SDF + 坐标)
2. ResNet-UNet 编码解码 → [B, hidden_channels, D', H', W']
3. grid_sample 在顶点位置采样 → [B, N, hidden_channels]
4. MLP 输出 → [B, N, 3]

## 依赖

- PyTorch
- VTK
- Open3D
- NumPy
- tqdm
- PyYAML
