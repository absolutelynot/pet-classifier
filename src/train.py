#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
训练脚本
只微调分类头，保存最佳模型
"""

import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import StepLR
from tqdm import tqdm
import argparse
from pathlib import Path

from model import create_model
from data_loader import get_data_loaders


def get_project_root():
    """获取项目根目录（pet-classifier目录）"""
    current_file = Path(__file__).resolve()
    # 从当前文件向上查找，直到找到包含 .git 或 requirements.txt 的目录
    for parent in current_file.parents:
        if (parent / 'requirements.txt').exists() or (parent / '.git').exists():
            return parent
    # 如果找不到，返回当前文件的上两级目录（假设在 src/ 下）
    return current_file.parent.parent


def train_epoch(model, train_loader, criterion, optimizer, device):
    """
    训练一个epoch
    
    Args:
        model: 模型
        train_loader: 训练数据加载器
        criterion: 损失函数
        optimizer: 优化器
        device: 设备
    
    Returns:
        average_loss: 平均损失
        accuracy: 准确率
    """
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0
    
    pbar = tqdm(train_loader, desc='Training')
    for images, labels in pbar:
        images = images.to(device)
        labels = labels.to(device)
        
        # 前向传播
        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        
        # 反向传播
        loss.backward()
        optimizer.step()
        
        # 统计
        running_loss += loss.item()
        _, predicted = torch.max(outputs.data, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()
        
        # 更新进度条
        pbar.set_postfix({
            'loss': f'{loss.item():.4f}',
            'acc': f'{100 * correct / total:.2f}%'
        })
    
    average_loss = running_loss / len(train_loader)
    accuracy = 100 * correct / total
    return average_loss, accuracy


def validate(model, val_loader, criterion, device):
    """
    验证模型
    
    Args:
        model: 模型
        val_loader: 验证数据加载器
        criterion: 损失函数
        device: 设备
    
    Returns:
        average_loss: 平均损失
        accuracy: 准确率
    """
    model.eval()
    running_loss = 0.0
    correct = 0
    total = 0
    
    with torch.no_grad():
        pbar = tqdm(val_loader, desc='Validating')
        for images, labels in pbar:
            images = images.to(device)
            labels = labels.to(device)
            
            outputs = model(images)
            loss = criterion(outputs, labels)
            
            running_loss += loss.item()
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
            
            pbar.set_postfix({
                'loss': f'{loss.item():.4f}',
                'acc': f'{100 * correct / total:.2f}%'
            })
    
    average_loss = running_loss / len(val_loader)
    accuracy = 100 * correct / total
    return average_loss, accuracy


def train(args):
    """
    主训练函数
    
    Args:
        args: 命令行参数
    """
    # 设备选择
    if torch.cuda.is_available():
        device = torch.device('cuda')
        print("使用CUDA")
    elif torch.backends.mps.is_available():
        device = torch.device('mps')
        print("使用MPS (Apple Silicon)")
    else:
        device = torch.device('cpu')
        print("使用CPU")
    
    # 如果model_dir是相对路径，转换为项目根目录下的绝对路径
    if args.model_dir is None or (not os.path.isabs(args.model_dir) and args.model_dir == 'models'):
        project_root = get_project_root()
        args.model_dir = str(project_root / 'models')
    
    # 创建保存目录
    os.makedirs(args.model_dir, exist_ok=True)
    print(f"模型保存目录: {args.model_dir}")
    
    # 加载数据
    print("加载数据...")
    train_loader, val_loader, test_loader, num_classes, classes = get_data_loaders(
        data_dir=args.data_dir,
        batch_size=args.batch_size,
        num_workers=args.num_workers
    )
    
    # 创建模型
    print("创建模型...")
    model = create_model(num_classes=num_classes, device=device)
    
    # 损失函数和优化器
    criterion = nn.CrossEntropyLoss()
    
    # 只优化分类头参数
    trainable_params = model.get_trainable_parameters()
    optimizer = optim.Adam(trainable_params, lr=args.learning_rate, weight_decay=args.weight_decay)
    scheduler = StepLR(optimizer, step_size=args.step_size, gamma=args.gamma)
    
    # 训练历史
    best_val_acc = 0.0
    train_losses = []
    train_accs = []
    val_losses = []
    val_accs = []
    
    print(f"\n开始训练，共{args.epochs}个epoch...")
    print(f"学习率: {args.learning_rate}")
    print(f"批次大小: {args.batch_size}")
    print("-" * 50)
    
    for epoch in range(1, args.epochs + 1):
        print(f"\nEpoch {epoch}/{args.epochs}")
        
        # 训练
        train_loss, train_acc = train_epoch(model, train_loader, criterion, optimizer, device)
        train_losses.append(train_loss)
        train_accs.append(train_acc)
        
        # 验证
        val_loss, val_acc = validate(model, val_loader, criterion, device)
        val_losses.append(val_loss)
        val_accs.append(val_acc)
        
        # 学习率调度
        scheduler.step()
        
        # 保存最佳模型
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            best_model_path = os.path.join(args.model_dir, 'best_model.pth')
            torch.save(model.state_dict(), best_model_path)
            print(f"✓ 保存最佳模型 (验证准确率: {val_acc:.2f}%)")
        
        # 打印epoch总结
        print(f"训练 - 损失: {train_loss:.4f}, 准确率: {train_acc:.2f}%")
        print(f"验证 - 损失: {val_loss:.4f}, 准确率: {val_acc:.2f}%")
        print(f"当前学习率: {scheduler.get_last_lr()[0]:.6f}")
    
    print("\n" + "=" * 50)
    print(f"训练完成！最佳验证准确率: {best_val_acc:.2f}%")
    print(f"最佳模型已保存至: {os.path.join(args.model_dir, 'best_model.pth')}")
    
    # 保存训练历史
    history = {
        'train_losses': train_losses,
        'train_accs': train_accs,
        'val_losses': val_losses,
        'val_accs': val_accs,
        'best_val_acc': best_val_acc
    }
    history_path = os.path.join(args.model_dir, 'training_history.pt')
    torch.save(history, history_path)
    print(f"训练历史已保存至: {history_path}")


def main():
    parser = argparse.ArgumentParser(description='训练宠物分类模型')
    parser.add_argument('--data-dir', type=str, default=None, help='数据目录（默认：项目根目录下的data/）')
    parser.add_argument('--model-dir', type=str, default=None, help='模型保存目录（默认：项目根目录下的models/）')
    parser.add_argument('--batch-size', type=int, default=32, help='批次大小')
    parser.add_argument('--epochs', type=int, default=20, help='训练轮数')
    parser.add_argument('--learning-rate', type=float, default=0.001, help='学习率')
    parser.add_argument('--weight-decay', type=float, default=1e-4, help='权重衰减')
    parser.add_argument('--step-size', type=int, default=7, help='学习率衰减步长')
    parser.add_argument('--gamma', type=float, default=0.1, help='学习率衰减系数')
    parser.add_argument('--num-workers', type=int, default=4, help='数据加载worker数量')
    
    args = parser.parse_args()
    train(args)


if __name__ == '__main__':
    main()
