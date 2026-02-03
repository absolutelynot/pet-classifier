#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
评估脚本
在测试集上评估模型，计算准确率、混淆矩阵等
"""

import os
import torch
import numpy as np
from sklearn.metrics import confusion_matrix, classification_report
import matplotlib.pyplot as plt
import seaborn as sns
import argparse
from tqdm import tqdm

from model import load_model
from data_loader import get_data_loaders, get_test_dataset


def evaluate_model(model, test_loader, device, class_names=None):
    """
    评估模型
    
    Args:
        model: 模型
        test_loader: 测试数据加载器
        device: 设备
        class_names: 类别名称列表
    
    Returns:
        accuracy: 准确率
        confusion_mat: 混淆矩阵
        predictions: 所有预测结果
        true_labels: 所有真实标签
        probabilities: 所有预测概率
    """
    model.eval()
    all_preds = []
    all_labels = []
    all_probs = []
    
    with torch.no_grad():
        pbar = tqdm(test_loader, desc='Evaluating')
        for images, labels in pbar:
            images = images.to(device)
            labels = labels.to(device)
            
            outputs = model(images)
            probs = torch.softmax(outputs, dim=1)
            _, predicted = torch.max(outputs, 1)
            
            all_preds.extend(predicted.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
            all_probs.extend(probs.cpu().numpy())
    
    # 计算准确率
    all_preds = np.array(all_preds)
    all_labels = np.array(all_labels)
    accuracy = 100 * np.sum(all_preds == all_labels) / len(all_labels)
    
    # 计算混淆矩阵
    confusion_mat = confusion_matrix(all_labels, all_preds)
    
    # 分类报告
    if class_names is None:
        class_names = [f'Class {i}' for i in range(len(confusion_mat))]
    
    report = classification_report(
        all_labels, 
        all_preds, 
        target_names=class_names,
        output_dict=True
    )
    
    return {
        'accuracy': accuracy,
        'confusion_matrix': confusion_mat,
        'predictions': all_preds,
        'true_labels': all_labels,
        'probabilities': np.array(all_probs),
        'classification_report': report,
        'class_names': class_names
    }


def plot_confusion_matrix(confusion_mat, class_names, save_path):
    """
    绘制混淆矩阵
    
    Args:
        confusion_mat: 混淆矩阵
        class_names: 类别名称列表
        save_path: 保存路径
    """
    plt.figure(figsize=(20, 16))
    
    # 归一化混淆矩阵（显示百分比）
    confusion_mat_norm = confusion_mat.astype('float') / confusion_mat.sum(axis=1)[:, np.newaxis]
    
    sns.heatmap(
        confusion_mat_norm,
        annot=True,
        fmt='.2f',
        cmap='Blues',
        xticklabels=class_names,
        yticklabels=class_names,
        cbar_kws={'label': 'Percentage'}
    )
    
    plt.title('Confusion Matrix (Normalized)', fontsize=16, pad=20)
    plt.ylabel('True Label', fontsize=14)
    plt.xlabel('Predicted Label', fontsize=14)
    plt.xticks(rotation=45, ha='right')
    plt.yticks(rotation=0)
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"混淆矩阵已保存至: {save_path}")


def evaluate(args):
    """
    主评估函数
    
    Args:
        args: 命令行参数
    """
    # 设备选择
    if torch.cuda.is_available():
        device = torch.device('cuda')
    elif torch.backends.mps.is_available():
        device = torch.device('mps')
    else:
        device = torch.device('cpu')
    
    print(f"使用设备: {device}")
    
    # 加载模型
    print(f"加载模型: {args.model_path}")
    model = load_model(args.model_path, num_classes=args.num_classes, device=device)
    
    # 加载测试数据
    if args.use_official_test:
        print("使用官方测试集...")
        test_loader, num_classes, class_names = get_test_dataset(
            data_dir=args.data_dir,
            batch_size=args.batch_size,
            num_workers=args.num_workers
        )
    else:
        print("使用划分的测试集...")
        _, _, test_loader, num_classes, class_names = get_data_loaders(
            data_dir=args.data_dir,
            batch_size=args.batch_size,
            num_workers=args.num_workers
        )
    
    # 评估
    print("开始评估...")
    results = evaluate_model(model, test_loader, device, class_names)
    
    # 打印结果
    print("\n" + "=" * 50)
    print("评估结果")
    print("=" * 50)
    print(f"总体准确率: {results['accuracy']:.2f}%")
    print("\n各类别准确率:")
    for i, class_name in enumerate(results['class_names']):
        if i in results['classification_report']:
            acc = results['classification_report'][i]['precision']
            print(f"  {class_name}: {acc:.2%}")
    
    # 保存混淆矩阵
    if args.save_confusion_matrix:
        os.makedirs(os.path.dirname(args.confusion_matrix_path), exist_ok=True)
        plot_confusion_matrix(
            results['confusion_matrix'],
            results['class_names'],
            args.confusion_matrix_path
        )
    
    # 保存评估结果
    if args.save_results:
        results_path = os.path.join(args.output_dir, 'evaluation_results.pt')
        os.makedirs(args.output_dir, exist_ok=True)
        # 只保存必要的数据（numpy数组可以保存）
        save_data = {
            'accuracy': results['accuracy'],
            'confusion_matrix': results['confusion_matrix'],
            'predictions': results['predictions'],
            'true_labels': results['true_labels'],
            'probabilities': results['probabilities'],
            'class_names': results['class_names'],
            'classification_report': results['classification_report']
        }
        torch.save(save_data, results_path)
        print(f"\n评估结果已保存至: {results_path}")
    
    return results


def main():
    parser = argparse.ArgumentParser(description='评估宠物分类模型')
    parser.add_argument('--model-path', type=str, required=True, help='模型权重路径')
    parser.add_argument('--data-dir', type=str, default='data', help='数据目录')
    parser.add_argument('--output-dir', type=str, default='models', help='输出目录')
    parser.add_argument('--batch-size', type=int, default=32, help='批次大小')
    parser.add_argument('--num-classes', type=int, default=37, help='类别数')
    parser.add_argument('--num-workers', type=int, default=4, help='数据加载worker数量')
    parser.add_argument('--use-official-test', action='store_true', help='使用官方测试集')
    parser.add_argument('--save-confusion-matrix', action='store_true', help='保存混淆矩阵')
    parser.add_argument('--confusion-matrix-path', type=str, default='models/confusion_matrix.png', help='混淆矩阵保存路径')
    parser.add_argument('--save-results', action='store_true', help='保存评估结果')
    
    args = parser.parse_args()
    evaluate(args)


if __name__ == '__main__':
    main()
