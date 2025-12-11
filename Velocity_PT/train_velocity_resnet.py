"""
速度场预测训练脚本
基于 ResNet-UNet 模型，预测三维速度场 (vx, vy, vz)
"""

import os
import sys

# 添加父目录到路径，以便导入共享模块
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# 获取脚本所在目录，用于构建输出路径
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
OUTPUT_DIR = os.path.join(SCRIPT_DIR, "output")

import argparse
import torch
import torch.nn as nn
import torch.nn.functional as F
import json
import random
import numpy as np
from pathlib import Path
from tqdm import tqdm
import yaml

# 导入数据模块
from data.velocity_datamodule_pt import instantiate_velocity_datamodule
from data.velocity_forwardset import loss_dict

# 导入速度场专用模型
from models.resnet_unet_velocity import ResNetUNetVelocity


class AverageMeter:
    """计算并存储平均值和当前值"""
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


def set_seed(seed=42):
    """设置随机种子以确保实验可重复性"""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


def parse_args(config_file_path=None):
    """解析命令行参数"""
    parser = argparse.ArgumentParser(description="训练和评估速度场预测模型")
    
    parser.add_argument('--config', type=str, default=config_file_path,
                        help='配置文件路径')
    parser.add_argument('--data_dir', type=str, help='数据目录')
    parser.add_argument('--index_file_dir', type=str, help='索引文件目录')
    parser.add_argument('--model_name', type=str, help='模型名称')
    parser.add_argument('--batch_size', type=int, help='训练批量大小')
    parser.add_argument('--eval_batch_size', type=int, help='评估批量大小')
    parser.add_argument('--lr', type=float, help='学习率')
    parser.add_argument('--num_epochs', type=int, help='训练轮数')
    parser.add_argument('--use_attention', type=lambda x: x.lower() == 'true',
                        help='是否使用注意力机制')
    parser.add_argument('--attention_type', type=str, choices=['cbam'],
                        help='注意力类型')
    parser.add_argument('--rel_weight', type=float, help='相对误差损失权重')
    parser.add_argument('--r2_weight', type=float, help='R2损失权重')
    parser.add_argument('--weight_decay', type=float, help='权重衰减')
    parser.add_argument('--track', type=str, default=None, help='输出目录标识')
    parser.add_argument('--normalizer_type', type=str, default='standard',
                        choices=['standard', 'minmax', 'robust', 'none'],
                        help='速度场标准化类型')
    parser.add_argument('--test_interval', type=int, default=10,
                        help='每隔多少个epoch测试一次模型')

    return parser.parse_args()


def load_config(config_path):
    """加载YAML配置文件"""
    try:
        print(f"正在加载配置文件: {config_path}")
        with open(config_path, 'r', encoding='utf-8') as f:
            config = yaml.safe_load(f)

        if config is None:
            raise ValueError(f"配置文件为空或格式错误: {config_path}")

        print("配置文件内容:")
        for key, value in config.items():
            print(f"  {key}: {value}")

        return config
    except Exception as e:
        print(f"\n配置文件加载错误: {config_path}")
        print(f"错误类型: {type(e).__name__}")
        print(f"错误信息: {str(e)}")
        raise


class VelocityLoss(nn.Module):
    """速度场损失函数"""
    def __init__(self, rel_weight=0.8, r2_weight=0.2, eps=1e-8):
        super().__init__()
        self.rel_weight = rel_weight
        self.r2_weight = r2_weight
        self.eps = eps
    
    def forward(self, pred, true):
        """
        计算速度场损失
        pred: [B, N, 3]
        true: [B, N, 3]
        """
        # 展平为 [B, N*3]
        pred_flat = pred.reshape(pred.shape[0], -1)
        true_flat = true.reshape(true.shape[0], -1)
        
        # 相对误差
        diff_norms = torch.norm(pred_flat - true_flat, p=2, dim=1)
        true_norms = torch.norm(true_flat, p=2, dim=1) + self.eps
        relative_errors = diff_norms / true_norms
        relative_errors = torch.clamp(relative_errors, max=100.0)
        mean_rel = torch.mean(relative_errors)
        
        # R²损失
        true_mean = torch.mean(true_flat, dim=1, keepdim=True)
        rss = torch.sum((true_flat - pred_flat) ** 2, dim=1)
        tss = torch.sum((true_flat - true_mean) ** 2, dim=1) + self.eps
        r2 = 1.0 - (rss / tss)
        r2_loss = 1.0 - r2
        r2_loss = torch.clamp(r2_loss, min=0.0, max=2.0)
        mean_r2_loss = torch.mean(r2_loss)
        
        # 组合损失
        combined_loss = self.rel_weight * mean_rel + self.r2_weight * mean_r2_loss
        
        return combined_loss, mean_rel.item(), mean_r2_loss.item()


def train_epoch(model, train_loader, optimizer, loss_fn, device):
    """训练一个epoch"""
    model.train()
    train_l2_meter = AverageMeter()
    rel_meter = AverageMeter()
    r2_meter = AverageMeter()
    mse_meter = AverageMeter()

    pbar = tqdm(train_loader, desc="训练中", leave=False)

    for i, data_dict in enumerate(pbar):
        x, output_points, true = loss_dict(data_dict)
        x = x.to(device)
        output_points = output_points.to(device)
        true = true.to(device)  # [B, N, 3]

        # 检查是否存在NaN或Inf
        x = torch.nan_to_num(x, nan=0.0, posinf=1.0, neginf=-1.0)
        true = torch.nan_to_num(true, nan=0.0, posinf=1.0, neginf=-1.0)

        # 前向传播，传入顶点坐标
        pred = model(x, output_points)  # [B, N, 3]

        # 检查预测值
        pred = torch.nan_to_num(pred, nan=0.0, posinf=1.0, neginf=-1.0)

        loss, rel, r2 = loss_fn(pred, true)

        # 计算MSE指标
        mse = F.mse_loss(pred, true).item()

        optimizer.zero_grad()
        loss.backward()

        # 梯度裁剪
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=0.5)

        optimizer.step()

        train_l2_meter.update(loss.item(), n=1)
        rel_meter.update(rel, n=1)
        r2_meter.update(r2, n=1)
        mse_meter.update(mse, n=1)

        pbar.set_postfix({
            'loss': f'{train_l2_meter.avg:.4f}',
            'rel': f'{rel_meter.avg:.4f}',
            'mse': f'{mse_meter.avg:.6f}'
        })

    metrics = {
        'loss': train_l2_meter.avg,
        'rel': rel_meter.avg,
        'r2': r2_meter.avg,
        'mse': mse_meter.avg
    }

    return metrics


def evaluate(model, datamodule, config, loss_fn, device, track="Velocity_Dataset"):
    """评估模型"""
    output_track_dir = os.path.join(OUTPUT_DIR, track)
    os.makedirs(output_track_dir, exist_ok=True)
    test_loader = datamodule.test_dataloader(
        batch_size=config["eval_batch_size"],
        shuffle=False,
        num_workers=0
    )

    model.eval()
    total_loss = 0
    total_rel = 0
    total_r2 = 0
    total_mse = 0
    total_rmse = 0
    num_batches = 0

    with torch.no_grad():
        test_pbar = tqdm(test_loader, desc="测试中")

        for i, data_dict in enumerate(test_pbar):
            x, output_points, true = loss_dict(data_dict)
            x = x.to(device)
            output_points = output_points.to(device)

            x = torch.nan_to_num(x, nan=0.0, posinf=1.0, neginf=-1.0)

            if true is not None:
                true = true.to(device)  # [B, N, 3]
                true = torch.nan_to_num(true, nan=0.0, posinf=1.0, neginf=-1.0)

            pred = model(x, output_points)  # [B, N, 3]
            pred = torch.nan_to_num(pred, nan=0.0, posinf=1.0, neginf=-1.0)

            loss_eval, rel, r2 = loss_fn(pred, true)

            mse = F.mse_loss(pred, true).item()
            rmse = np.sqrt(mse)

            if not np.isnan(loss_eval.item()) and not np.isinf(loss_eval.item()):
                total_loss += loss_eval.item()
                total_rel += rel
                total_r2 += r2
                total_mse += mse
                total_rmse += rmse
                num_batches += 1

                if num_batches > 0:
                    test_pbar.set_postfix({
                        'loss': f'{total_loss / num_batches:.4f}',
                        'rel': f'{total_rel / num_batches:.4f}',
                        'mse': f'{total_mse / num_batches:.6f}'
                    })

            # 保存预测结果（反标准化后）
            pred_decoded = datamodule.decode(pred.cpu())
            test_idx = datamodule.test_indices[i]
            save_path = os.path.join(output_track_dir, f"vel_pred_{str(test_idx).zfill(3)}.npy")
            np.save(save_path, pred_decoded.numpy())

    n = max(num_batches, 1)
    metrics = {
        'loss': total_loss / n,
        'rel': total_rel / n,
        'r2': total_r2 / n,
        'mse': total_mse / n,
        'rmse': total_rmse / n
    }
    return metrics


def save_checkpoint(model, optimizer, scheduler, epoch, metrics, path):
    """保存检查点"""
    torch.save({
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'scheduler_state_dict': scheduler.state_dict(),
        'train_metrics': metrics
    }, path)
    print(f"模型已保存到: {path}")


def print_metrics(metrics, prefix=""):
    """打印评估指标"""
    print(f"{prefix}指标摘要:")
    if 'loss' in metrics:
        print(f"{prefix}  总损失(Loss): {metrics['loss']:.6f}")
    if 'rel' in metrics:
        print(f"{prefix}  相对误差(Rel): {metrics['rel']:.6f}")
    if 'r2' in metrics:
        print(f"{prefix}  R²损失: {metrics['r2']:.6f}")
    if 'mse' in metrics:
        print(f"{prefix}  均方误差(MSE): {metrics['mse']:.6f}")
    if 'rmse' in metrics:
        print(f"{prefix}  均方根误差(RMSE): {metrics['rmse']:.6f}")


def main():
    # 设置随机种子
    set_seed(42)

    # 设置设备
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"使用设备: {device}")

    # 清理CUDA缓存
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    # 解析命令行参数
    args = parse_args("Velocity_PT/configs/velocity_resnet.yaml")

    # 加载配置
    config = load_config(args.config)

    # 更新配置
    for key, value in vars(args).items():
        if key != "config" and value is not None:
            config[key] = value

    test_interval = config.get("test_interval", 10)

    # 创建输出目录
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    # 初始化训练历史记录
    training_history = []

    history_file = os.path.join(OUTPUT_DIR, f"training_history_{config['model_name']}.json")
    print(f"训练历史将保存到: {history_file}")

    # 初始化数据模块
    print("\n初始化数据模块...")
    datamodule = instantiate_velocity_datamodule(config)
    train_loader = datamodule.train_dataloader(batch_size=config["batch_size"], shuffle=True)
    test_loader = datamodule.test_dataloader(batch_size=config["eval_batch_size"], shuffle=False)

    # 获取样本批次以确定输入/输出形状
    print("\n获取样本批次...")
    sample_batch = next(iter(train_loader))
    x, output_points, true = loss_dict(sample_batch)
    input_channels = x.shape[1]
    num_vertices = output_points.shape[1]
    out_channels = true.shape[2] if true is not None else 3  # 速度有三个分量

    print(f"输入形状: {x.shape}")
    print(f"顶点坐标形状: {output_points.shape}")
    print(f"速度场形状: {true.shape if true is not None else 'None'}")

    # 初始化速度场专用模型
    use_attention = config.get("use_attention", True)
    hidden_channels = config.get("hidden_channels", 48)

    model = ResNetUNetVelocity(
        in_channels=input_channels,
        out_channels=out_channels,
        hidden_channels=hidden_channels,
        use_attention=use_attention
    ).to(device)

    print(f"使用ResNet-UNet速度场预测模型")
    print(f"模型配置: 输入通道={input_channels}, 隐藏通道={hidden_channels}, 输出通道={out_channels}")
    print(f"顶点数量: {num_vertices}")
    
    # 计算模型参数量
    total_params = sum(p.numel() for p in model.parameters())
    print(f"模型参数量: {total_params:,}")

    # 初始化优化器
    weight_decay = config.get("weight_decay", 0.02)
    beta1 = config.get("beta1", 0.9)
    beta2 = config.get("beta2", 0.999)
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=config["lr"],
        weight_decay=weight_decay,
        betas=(beta1, beta2)
    )

    # 初始化学习率调度器
    t0 = config.get("t0", 8)
    t_mult = config.get("t_mult", 1)
    eta_min_factor = config.get("eta_min_factor", 0.001)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(
        optimizer,
        T_0=t0,
        T_mult=t_mult,
        eta_min=config["lr"] * eta_min_factor
    )

    # 初始化损失函数
    rel_weight = config.get("rel_weight", 0.8)
    r2_weight = config.get("r2_weight", 0.2)

    loss_fn = VelocityLoss(
        rel_weight=rel_weight,
        r2_weight=r2_weight
    )

    print(f"损失函数权重: rel_weight={rel_weight}, r2_weight={r2_weight}")
    print(f"优化器参数: beta1={beta1}, beta2={beta2}")
    print(f"学习率调度器: T_0={t0}, T_mult={t_mult}, eta_min_factor={eta_min_factor}")

    # 训练参数
    best_train_loss = float('inf')
    best_test_loss = float('inf')
    best_epoch = -1
    best_test_metrics = None
    warmup_epochs = config.get("warmup_epochs", 8)

    print(f"\n开始训练 {config['model_name']} 模型，共{config['num_epochs']}个epoch")
    print(f"预热轮数: {warmup_epochs}, 每 {test_interval} 个epoch在测试集上评估一次模型")

    epoch_pbar = tqdm(range(config["num_epochs"]), desc="总进度")

    for ep in epoch_pbar:
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        # 学习率预热
        if ep < warmup_epochs:
            for param_group in optimizer.param_groups:
                param_group['lr'] = config["lr"] * (ep + 1) / warmup_epochs

        # 训练一个epoch
        train_metrics = train_epoch(model, train_loader, optimizer, loss_fn, device)

        # 更新学习率
        if ep >= warmup_epochs:
            scheduler.step()

        # 打印训练指标
        print(f"\nEpoch {ep}/{config['num_epochs'] - 1}")
        print(f"学习率: {optimizer.param_groups[0]['lr']:.6f}")
        print(f"训练集损失:")
        print_metrics(train_metrics)

        epoch_pbar.set_postfix({
            'train_loss': f'{train_metrics["loss"]:.4f}',
            'lr': f'{optimizer.param_groups[0]["lr"]:.6f}'
        })

        # 记录训练历史
        history_entry = {
            'epoch': ep,
            'train_loss': train_metrics['loss'],
            'train_rel': train_metrics['rel'],
            'train_r2': train_metrics['r2'],
            'train_mse': train_metrics.get('mse', 0.0),
            'lr': optimizer.param_groups[0]['lr']
        }
        training_history.append(history_entry)

        # 定期评估
        if (ep + 1) % test_interval == 0 or ep == config['num_epochs'] - 1:
            print(f"\n在测试集上评估模型 (Epoch {ep})...")
            test_metrics = evaluate(model, datamodule, config, loss_fn, device, config["track"])
            print(f"测试集上的损失值：")
            print_metrics(test_metrics)

            if test_metrics['loss'] < best_test_loss:
                best_test_loss = test_metrics['loss']
                best_train_loss = train_metrics['loss']
                best_epoch = ep
                best_test_metrics = test_metrics

                save_path = os.path.join(OUTPUT_DIR, f"best_model-{config['model_name']}.pth")
                save_checkpoint(model, optimizer, scheduler, ep, test_metrics, save_path)
                print(f"\n发现更好的模型！保存到: {save_path}")
                print(f"当前最佳测试损失: {best_test_loss:.6f} (Epoch {best_epoch})")

            test_entry = {
                'epoch': f'test_at_epoch_{ep}',
                'test_loss': test_metrics['loss'],
                'test_rel': test_metrics['rel'],
                'test_r2': test_metrics['r2'],
                'test_mse': test_metrics['mse'],
                'test_rmse': test_metrics['rmse']
            }
            training_history.append(test_entry)

        # 保存训练历史
        try:
            with open(history_file, 'w') as f:
                json.dump(training_history, f, indent=4)
        except Exception as e:
            print(f"保存训练历史时出错: {e}")

    print(f"\n训练完成！")
    print(f"最佳模型在第 {best_epoch} 个epoch")
    model_name = config['model_name']
    print(f"最佳模型保存在: {os.path.join(OUTPUT_DIR, f'best_model-{model_name}.pth')}")

    if best_test_metrics:
        print("\n最佳模型在测试集上的结果:")
        print_metrics(best_test_metrics)


if __name__ == "__main__":
    main()
