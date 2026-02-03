#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
数据加载和预处理模块
下载Oxford-IIIT Pet数据集，进行预处理，创建DataLoader
"""

import os
import torch
from torch.utils.data import DataLoader, random_split
from torchvision import datasets, transforms
from torchvision.datasets import OxfordIIITPet


def get_data_loaders(data_dir='data', batch_size=32, num_workers=4, val_split=0.15, test_split=0.15):
    """
    下载Oxford-IIIT Pet数据集并创建DataLoader
    
    Args:
        data_dir: 数据存储目录
        batch_size: 批次大小
        num_workers: 数据加载的worker数量
        val_split: 验证集比例
        test_split: 测试集比例
    
    Returns:
        train_loader, val_loader, test_loader, num_classes
    """
    # 数据目录
    data_path = os.path.join(data_dir, 'oxford-iiit-pet')
    os.makedirs(data_path, exist_ok=True)
    
    # 数据预处理
    # 训练集：数据增强
    train_transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.RandomRotation(degrees=15),
        transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], 
                           std=[0.229, 0.224, 0.225])
    ])
    
    # 验证集和测试集：只做resize和normalize
    val_test_transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], 
                           std=[0.229, 0.224, 0.225])
    ])
    
    # 下载完整数据集（用于后续划分）
    full_dataset = OxfordIIITPet(
        root=data_path,
        split='trainval',
        target_types='category',
        download=True,
        transform=train_transform  # 先用train_transform，后面会重新创建
    )
    
    # 获取类别数量（Oxford-IIIT Pet有37个类别）
    num_classes = len(full_dataset.classes)
    print(f"数据集类别数: {num_classes}")
    print(f"数据集总样本数: {len(full_dataset)}")
    
    # 划分数据集：训练集、验证集、测试集
    total_size = len(full_dataset)
    test_size = int(total_size * test_split)
    val_size = int(total_size * val_split)
    train_size = total_size - val_size - test_size
    
    # 使用随机种子确保可复现
    generator = torch.Generator().manual_seed(42)
    train_dataset, val_dataset, test_dataset = random_split(
        full_dataset, 
        [train_size, val_size, test_size],
        generator=generator
    )
    
    # 为验证集和测试集重新创建，使用val_test_transform
    # 注意：random_split会保留原始transform，我们需要手动更新
    val_indices = val_dataset.indices
    test_indices = test_dataset.indices
    
    # 重新创建验证集和测试集，使用不同的transform
    val_dataset_full = OxfordIIITPet(
        root=data_path,
        split='trainval',
        target_types='category',
        download=False,
        transform=val_test_transform
    )
    test_dataset_full = OxfordIIITPet(
        root=data_path,
        split='trainval',
        target_types='category',
        download=False,
        transform=val_test_transform
    )
    
    # 创建子集
    from torch.utils.data import Subset
    val_dataset = Subset(val_dataset_full, val_indices)
    test_dataset = Subset(test_dataset_full, test_indices)
    
    # 创建DataLoader
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True if torch.cuda.is_available() else False
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True if torch.cuda.is_available() else False
    )
    
    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True if torch.cuda.is_available() else False
    )
    
    print(f"训练集样本数: {len(train_dataset)}")
    print(f"验证集样本数: {len(val_dataset)}")
    print(f"测试集样本数: {len(test_dataset)}")
    
    return train_loader, val_loader, test_loader, num_classes, full_dataset.classes


def get_test_dataset(data_dir='data', batch_size=32, num_workers=4):
    """
    获取官方测试集（如果可用）
    
    Args:
        data_dir: 数据存储目录
        batch_size: 批次大小
        num_workers: 数据加载的worker数量
    
    Returns:
        test_loader, num_classes
    """
    data_path = os.path.join(data_dir, 'oxford-iiit-pet')
    
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], 
                           std=[0.229, 0.224, 0.225])
    ])
    
    # Oxford-IIIT Pet的官方测试集
    test_dataset = OxfordIIITPet(
        root=data_path,
        split='test',
        target_types='category',
        download=True,
        transform=transform
    )
    
    num_classes = len(test_dataset.classes)
    
    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True if torch.cuda.is_available() else False
    )
    
    return test_loader, num_classes, test_dataset.classes


if __name__ == '__main__':
    # 测试数据加载
    print("测试数据加载...")
    train_loader, val_loader, test_loader, num_classes, classes = get_data_loaders(
        data_dir='data',
        batch_size=32,
        num_workers=2
    )
    print(f"类别列表: {classes}")
    print("数据加载测试完成！")
