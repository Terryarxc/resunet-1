"""
速度场预测数据模块 (PyTorch版本)
从 .vtk 文件读取速度场数据 (point_vectors)，配合SDF体素输入
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

import vtk
import torch
import numpy as np
from pathlib import Path
from vtk.util.numpy_support import vtk_to_numpy
from torch.utils.data import Dataset, DataLoader
from typing import List, Tuple, Union
import warnings
import open3d as o3d


class UnitGaussianNormalizer:
    """标准化器，用于速度场的归一化处理"""
    def __init__(self, x, eps=1e-05, reduce_dim=(0, 1), verbose=True):
        n_samples, *shape = x.shape
        self.sample_shape = shape
        self.verbose = verbose
        self.reduce_dim = reduce_dim
        y = x.numpy() if isinstance(x, torch.Tensor) else x
        self.mean = torch.from_numpy(
            np.mean(y, axis=reduce_dim, keepdims=True).squeeze(axis=0)
        ).float()
        self.std = torch.from_numpy(
            np.std(y, axis=reduce_dim, keepdims=True).squeeze(axis=0)
        ).float()
        self.eps = eps

    def encode(self, x):
        x = x - self.mean.to(x.device)
        x = x / (self.std.to(x.device) + self.eps)
        return x

    def decode(self, x, sample_idx=None):
        std = self.std + self.eps
        mean = self.mean
        if sample_idx is not None:
            if len(self.mean.shape) == len(sample_idx[0].shape):
                std = self.std[sample_idx] + self.eps
                mean = self.mean[sample_idx]
            elif len(self.mean.shape) > len(sample_idx[0].shape):
                std = self.std[:, sample_idx] + self.eps
                mean = self.mean[:, sample_idx]
        x = x * std.to(x.device)
        x = x + mean.to(x.device)
        return x

    def cuda(self):
        self.mean = self.mean.cuda()
        self.std = self.std.cuda()
        return self

    def cpu(self):
        self.mean = self.mean.cpu()
        self.std = self.std.cpu()
        return self


def read_vtk(file_path):
    """读取VTK文件"""
    reader = vtk.vtkUnstructuredGridReader()
    reader.SetFileName(str(file_path))
    reader.Update()
    polydata = reader.GetOutput()
    return reader, polydata


def nodes(polydata):
    """获取顶点坐标"""
    points = vtk_to_numpy(polydata.GetPoints().GetData()).astype(np.float32)
    return points


def velocity(polydata):
    """获取速度场数据 (point_vectors)"""
    vel = vtk_to_numpy(polydata.GetPointData().GetArray("point_vectors")).astype(np.float32)
    return vel


def get_triangles_from_vtk(polydata):
    """从VTK polydata中提取三角形面片索引
    
    支持三角形和四边形网格，四边形会被拆分成2个三角形
    """
    cells = polydata.GetCells()
    cell_array = vtk_to_numpy(cells.GetData())
    
    triangles = []
    i = 0
    while i < len(cell_array):
        n_points = cell_array[i]
        if n_points == 3:  # 三角形
            triangles.append(cell_array[i+1:i+4])
        elif n_points == 4:  # 四边形，拆分成2个三角形
            v0, v1, v2, v3 = cell_array[i+1:i+5]
            triangles.append([v0, v1, v2])
            triangles.append([v0, v2, v3])
        i += n_points + 1
    
    return np.array(triangles, dtype=np.int32)


class DictDataset(Dataset):
    """字典数据集"""
    def __init__(self, data_dict: dict):
        self.data_dict = data_dict
        lengths = [len(v) for v in data_dict.values()]
        assert all(l == lengths[0] for l in lengths), "All data must have the same length"

    def __getitem__(self, index):
        return {k: v[index] for k, v in self.data_dict.items()}

    def __len__(self):
        return len(next(iter(self.data_dict.values())))


class DictDatasetWithConstant(DictDataset):
    """带常量的字典数据集"""
    def __init__(self, data_dict: dict, constant_dict: dict):
        super().__init__(data_dict)
        self.constant_dict = constant_dict

    def __getitem__(self, index):
        return_dict = super().__getitem__(index)
        return_dict.update(self.constant_dict)
        return return_dict


class BaseDataModule:
    """数据模块基类"""
    @property
    def train_dataset(self) -> Dataset:
        raise NotImplementedError

    @property
    def test_dataset(self) -> Dataset:
        raise NotImplementedError

    def train_dataloader(self, **kwargs) -> DataLoader:
        collate_fn = getattr(self, 'collate_fn', None)
        return DataLoader(self.train_data, collate_fn=collate_fn, **kwargs)

    def test_dataloader(self, **kwargs) -> DataLoader:
        collate_fn = getattr(self, 'collate_fn', None)
        return DataLoader(self.test_data, collate_fn=collate_fn, **kwargs)


class VelocitySDFDataModule(BaseDataModule):
    """
    速度场SDF数据模块
    读取vtk文件中的速度场数据，配合SDF体素输入
    """
    def __init__(
        self,
        data_dir,
        index_file_dir,
        n_train: int = 50,
        n_test: int = 10,
        spatial_resolution: Tuple[int, int, int] = (64, 64, 64),
        eps=0.01,
        normalizer_type: str = "standard",
    ):
        BaseDataModule.__init__(self)
        
        self.normalizer_type = normalizer_type
        
        if isinstance(data_dir, str):
            data_dir = Path(data_dir)
        data_dir = data_dir.expanduser()
        
        if isinstance(index_file_dir, str):
            index_file_dir = Path(index_file_dir)
        index_file_dir = index_file_dir.expanduser()
        
        print(f'速度场数据目录: {data_dir}')
        print(f'索引文件目录: {index_file_dir}')
        
        assert data_dir.exists(), f"数据目录不存在: {data_dir}"
        assert data_dir.is_dir(), f"路径不是目录: {data_dir}"
        
        self.data_dir = data_dir
        self.index_file_dir = index_file_dir
        
        # 加载边界信息
        min_bounds, max_bounds = self.load_bound(index_file_dir, eps=eps)
        
        # 加载有效索引
        valid_mesh_inds = self.load_valid_mesh_indices(index_file_dir)
        print(f"有效索引数量: {len(valid_mesh_inds)}")
        
        assert n_train + n_test <= len(valid_mesh_inds), \
            f"请求的数据量({n_train + n_test})超过可用数据量({len(valid_mesh_inds)})"
        
        # 划分训练集和测试集
        train_indices = valid_mesh_inds[:n_train]
        test_indices = valid_mesh_inds[n_train:n_train + n_test]
        
        print(f"\n数据集划分:")
        print(f"训练集样本数: {len(train_indices)}")
        print(f"测试集样本数: {len(test_indices)}")
        
        # 获取vtk文件路径（流场数据和汽车表面网格）
        train_vtk_paths = [self.get_vtk_path(data_dir, i) for i in train_indices]
        test_vtk_paths = [self.get_vtk_path(data_dir, i) for i in test_indices]
        train_mesh_paths = [self.get_mesh_vtk_path(data_dir, i) for i in train_indices]
        test_mesh_paths = [self.get_mesh_vtk_path(data_dir, i) for i in test_indices]
        
        self.test_vtk_paths = test_vtk_paths
        self.test_indices = test_indices
        
        # 创建查询点网格
        tx = np.linspace(min_bounds[0], max_bounds[0], spatial_resolution[0])
        ty = np.linspace(min_bounds[1], max_bounds[1], spatial_resolution[1])
        tz = np.linspace(min_bounds[2], max_bounds[2], spatial_resolution[2])
        query_points = np.stack(
            np.meshgrid(tx, ty, tz, indexing="ij"), axis=-1
        ).astype(np.float32)
        print(f"查询点形状: {query_points.shape}")
        
        # 处理训练数据
        print("\n处理训练数据...")
        train_data_list = [
            self.load_vtk_data(vtk_path, mesh_path, query_points)
            for vtk_path, mesh_path in zip(train_vtk_paths, train_mesh_paths)
        ]
        train_sdf = torch.stack([torch.tensor(d['sdf']) for d in train_data_list])
        train_direction = torch.stack([torch.tensor(d['direction']) for d in train_data_list])
        train_vertices = torch.stack([torch.tensor(d['vertices']) for d in train_data_list])
        train_velocity = torch.stack([torch.tensor(d['velocity']) for d in train_data_list])
        del train_data_list
        
        # 处理测试数据
        print("\n处理测试数据...")
        test_data_list = [
            self.load_vtk_data(vtk_path, mesh_path, query_points)
            for vtk_path, mesh_path in zip(test_vtk_paths, test_mesh_paths)
        ]
        test_sdf = torch.stack([torch.tensor(d['sdf']) for d in test_data_list])
        test_direction = torch.stack([torch.tensor(d['direction']) for d in test_data_list])
        test_vertices = torch.stack([torch.tensor(d['vertices']) for d in test_data_list])
        test_velocity = torch.stack([torch.tensor(d['velocity']) for d in test_data_list])
        del test_data_list
        
        # 边界归一化
        min_bounds_t = torch.tensor(min_bounds)
        max_bounds_t = torch.tensor(max_bounds)
        train_vertices = self.location_normalization(train_vertices, min_bounds_t, max_bounds_t)
        test_vertices = self.location_normalization(test_vertices, min_bounds_t, max_bounds_t)
        
        # 速度标准化 - 保持 [N, 3] 形状，不展平
        print(f"\n速度场形状: 训练集={train_velocity.shape}, 测试集={test_velocity.shape}")
        
        # 对速度进行标准化（按所有样本的所有点计算均值和标准差）
        # 先展平用于计算统计量，但存储时保持原形状
        all_train_vel = train_velocity.reshape(-1, 3)  # [N_total, 3]
        self.vel_mean = all_train_vel.mean(dim=0)  # [3]
        self.vel_std = all_train_vel.std(dim=0) + 1e-6  # [3]
        
        print(f"速度均值: {self.vel_mean}")
        print(f"速度标准差: {self.vel_std}")
        
        # 标准化速度
        train_velocity_norm = (train_velocity - self.vel_mean) / self.vel_std
        test_velocity_norm = (test_velocity - self.vel_mean) / self.vel_std
        
        # 查询点归一化
        normalized_query_points = self.location_normalization(
            torch.tensor(query_points), min_bounds_t, max_bounds_t
        ).permute(3, 0, 1, 2)
        
        # 方向向量转换为 [3, D, H, W] 格式
        train_direction = train_direction.permute(0, 4, 1, 2, 3)  # [B, 3, D, H, W]
        test_direction = test_direction.permute(0, 4, 1, 2, 3)
        
        print(f"\n方向向量形状: 训练集={train_direction.shape}, 测试集={test_direction.shape}")
        
        # 构建数据集 - 速度保持 [N, 3] 形状
        self._train_data = DictDatasetWithConstant(
            {
                "sdf": train_sdf,
                "direction": train_direction,  # 新增方向向量
                "vertices": train_vertices,
                "velocity": train_velocity_norm
            },
            {"sdf_query_points": normalized_query_points}
        )
        
        self._test_data = DictDatasetWithConstant(
            {
                "sdf": test_sdf,
                "direction": test_direction,  # 新增方向向量
                "vertices": test_vertices,
                "velocity": test_velocity_norm
            },
            {"sdf_query_points": normalized_query_points}
        )
        self.min_bounds = min_bounds
        self.max_bounds = max_bounds
    
    @property
    def train_data(self):
        return self._train_data
    
    @property
    def test_data(self):
        return self._test_data
    
    def get_vtk_path(self, data_dir: Path, mesh_ind: int) -> Path:
        """获取流场vtk文件路径 (vel_###.vtk)"""
        vtk_path = data_dir / f"vel_{str(mesh_ind).zfill(3)}.vtk"
        if not vtk_path.exists():
            raise FileNotFoundError(f"流场VTK文件不存在: {vtk_path}")
        return vtk_path
    
    def get_mesh_vtk_path(self, data_dir: Path, mesh_ind: int) -> Path:
        """获取汽车表面网格vtk文件路径 (mesh_###.vtk)"""
        mesh_path = data_dir / f"mesh_{str(mesh_ind).zfill(3)}.vtk"
        if not mesh_path.exists():
            raise FileNotFoundError(f"汽车表面VTK文件不存在: {mesh_path}")
        return mesh_path
    
    def load_vtk_data(self, vtk_path: Path, mesh_vtk_path: Path, query_points: np.ndarray) -> dict:
        """加载VTK文件数据
        
        Args:
            vtk_path: 流场数据文件 (vel_###.vtk)
            mesh_vtk_path: 汽车表面网格文件 (mesh_###.vtk)
            query_points: SDF查询点网格
        """
        # 读取流场数据
        _, vel_polydata = read_vtk(str(vtk_path))
        vertices = nodes(vel_polydata)
        vel = velocity(vel_polydata)
        
        # 读取汽车表面网格
        _, mesh_polydata = read_vtk(str(mesh_vtk_path))
        mesh_vertices = nodes(mesh_polydata)
        mesh_triangles = get_triangles_from_vtk(mesh_polydata)
        
        print(f"加载 {vtk_path.name}: 流场点数={vertices.shape[0]}, 速度形状={vel.shape}")
        print(f"加载 {mesh_vtk_path.name}: 表面顶点数={mesh_vertices.shape[0]}, 三角形数={mesh_triangles.shape[0]}")
        
        # 使用汽车表面网格计算SDF和方向向量
        sdf, direction = self.compute_sdf_and_direction(mesh_vertices, mesh_triangles, query_points)
        
        return {
            'vertices': vertices,
            'velocity': vel,
            'sdf': sdf,
            'direction': direction  # 新增方向向量
        }
    
    def compute_sdf_and_direction(self, vertices: np.ndarray, triangles: np.ndarray, query_points: np.ndarray) -> tuple:
        """
        从汽车表面三角形网格计算SDF和方向向量
        
        Args:
            vertices: 汽车表面顶点 [N, 3]
            triangles: 三角形面片索引 [M, 3]
            query_points: 查询点网格 [64, 64, 64, 3]
        
        Returns:
            sdf: 有符号距离场 [64, 64, 64]
            direction: 方向向量（指向最近表面点） [64, 64, 64, 3]
        """
        # 创建Open3D三角形网格
        mesh = o3d.geometry.TriangleMesh()
        mesh.vertices = o3d.utility.Vector3dVector(vertices.astype(np.float64))
        mesh.triangles = o3d.utility.Vector3iVector(triangles)
        
        # 转换为tensor格式用于光线投射
        mesh_t = o3d.t.geometry.TriangleMesh.from_legacy(mesh)
        
        # 创建光线投射场景
        scene = o3d.t.geometry.RaycastingScene()
        _ = scene.add_triangles(mesh_t)
        
        query_flat = query_points.reshape(-1, 3).astype(np.float32)
        query_tensor = o3d.core.Tensor(query_flat, dtype=o3d.core.Dtype.Float32)
        
        # 计算有符号距离
        signed_distance = scene.compute_signed_distance(query_tensor).numpy()
        
        # 计算最近点，用于获取方向向量
        closest_points = scene.compute_closest_points(query_tensor)['points'].numpy()
        
        # 计算方向向量：从查询点指向最近表面点
        direction = closest_points - query_flat
        # 归一化方向向量
        direction_norm = np.linalg.norm(direction, axis=1, keepdims=True) + 1e-8
        direction = direction / direction_norm
        
        # 重塑为网格形状
        sdf = signed_distance.reshape(query_points.shape[:-1]).astype(np.float32)
        direction = direction.reshape(query_points.shape).astype(np.float32)
        
        return sdf, direction
    
    def load_bound(self, data_dir, filename="watertight_global_bounds.txt", eps=1e-06):
        """加载边界信息"""
        bound_file = data_dir / filename
        if not bound_file.exists():
            raise FileNotFoundError(f"边界文件不存在: {bound_file}")
        
        with open(bound_file, "r") as fp:
            min_bounds = fp.readline().split(" ")
            max_bounds = fp.readline().split(" ")
            min_bounds = [(float(a) - eps) for a in min_bounds]
            max_bounds = [(float(a) + eps) for a in max_bounds]
        return min_bounds, max_bounds
    
    def load_valid_mesh_indices(self, data_dir, filename="watertight_meshes.txt") -> List[int]:
        """加载有效的网格索引"""
        text_file_path = data_dir / filename
        
        if text_file_path.exists():
            try:
                with open(text_file_path, "r") as fp:
                    indices = [int(line) for line in fp.read().splitlines() if line.strip()]
                    print(f"从{filename}读取到{len(indices)}个有效索引")
                    return indices
            except Exception as e:
                print(f"读取{filename}出错: {e}")
        
        raise FileNotFoundError(f"索引文件不存在: {text_file_path}")
    
    def location_normalization(
        self,
        locations: torch.Tensor,
        min_bounds: torch.Tensor,
        max_bounds: torch.Tensor,
    ) -> torch.Tensor:
        """位置归一化到 [-1, 1]"""
        locations = (locations - min_bounds) / (max_bounds - min_bounds)
        return 2 * locations - 1
    
    def encode(self, velocity: torch.Tensor) -> torch.Tensor:
        """标准化速度"""
        return (velocity - self.vel_mean.to(velocity.device)) / self.vel_std.to(velocity.device)
    
    def decode(self, output: torch.Tensor) -> torch.Tensor:
        """反标准化速度"""
        return output * self.vel_std.to(output.device) + self.vel_mean.to(output.device)


def instantiate_velocity_datamodule(config):
    """实例化速度场数据模块"""
    return VelocitySDFDataModule(
        data_dir=config["data_dir"],
        index_file_dir=config["index_file_dir"],
        n_train=config["n_train"],
        n_test=config["n_test"],
        spatial_resolution=config["sdf_spatial_resolution"],
        normalizer_type=config.get("normalizer_type", "standard"),
    )


if __name__ == "__main__":
    # 测试数据模块
    config = {
        "data_dir": "../data/train_velocity_vtk",
        "index_file_dir": "../data/train_velocity",
        "n_train": 5,
        "n_test": 2,
        "sdf_spatial_resolution": [64, 64, 64],
        "normalizer_type": "standard"
    }
    
    datamodule = instantiate_velocity_datamodule(config)
    train_loader = datamodule.train_dataloader(batch_size=1, shuffle=False)
    
    for batch in train_loader:
        print(f"SDF形状: {batch['sdf'].shape}")
        print(f"顶点形状: {batch['vertices'].shape}")
        print(f"速度形状: {batch['velocity'].shape}")
        print(f"查询点形状: {batch['sdf_query_points'].shape}")
        break
