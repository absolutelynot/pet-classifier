#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
错误分析脚本
找出模型最容易混淆的5对类别，并输出各自的top错误图片
"""

import os
import torch
import numpy as np
import argparse
from collections import defaultdict
from PIL import Image
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
import sys
from pathlib import Path

# 添加src目录到路径
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from model import load_model
from data_loader import get_data_loaders, get_test_dataset
from evaluate import evaluate_model


def get_project_root():
    """获取项目根目录（pet-classifier目录）"""
    current_file = Path(__file__).resolve()
    for parent in current_file.parents:
        if (parent / 'requirements.txt').exists() or (parent / '.git').exists():
            return parent
    return current_file.parent.parent


def find_confused_pairs(confusion_matrix, class_names, top_k=5):
    """
    找出最容易混淆的类别对
    
    Args:
        confusion_matrix: 混淆矩阵
        class_names: 类别名称列表
        top_k: 返回前k对
    
    Returns:
        confused_pairs: [(true_class, pred_class, count), ...] 按count降序排列
    """
    confused_pairs = []
    n_classes = len(confusion_matrix)
    
    # 遍历混淆矩阵，找出非对角元素（错误分类）
    for i in range(n_classes):
        for j in range(n_classes):
            if i != j and confusion_matrix[i, j] > 0:
                confused_pairs.append((
                    i,  # 真实类别
                    j,  # 预测类别
                    confusion_matrix[i, j]  # 错误次数
                ))
    
    # 按错误次数降序排序
    confused_pairs.sort(key=lambda x: x[2], reverse=True)
    
    # 返回前top_k对
    top_pairs = confused_pairs[:top_k]
    
    # 格式化输出
    result = []
    for true_idx, pred_idx, count in top_pairs:
        result.append({
            'true_class': true_idx,
            'pred_class': pred_idx,
            'true_name': class_names[true_idx],
            'pred_name': class_names[pred_idx],
            'count': count
        })
    
    return result


def get_error_samples(model, test_loader, device, true_class, pred_class, class_names, top_k=10):
    """
    获取特定错误类型的样本（真实类别为true_class，但预测为pred_class）
    按置信度排序，返回top_k个
    
    Args:
        model: 模型
        test_loader: 测试数据加载器
        device: 设备
        true_class: 真实类别索引
        pred_class: 预测类别索引
        class_names: 类别名称列表
        top_k: 返回前k个样本
    
    Returns:
        error_samples: [(image_path, image_tensor, confidence, true_label, pred_label), ...]
    """
    model.eval()
    error_samples = []
    
    # 需要访问原始数据集来获取图片路径
    # 这里我们保存图片的索引和相关信息
    sample_idx = 0
    
    with torch.no_grad():
        for images, labels in test_loader:
            images_batch = images.to(device)
            labels_batch = labels.to(device)
            
            outputs = model(images_batch)
            probs = torch.softmax(outputs, dim=1)
            _, predicted = torch.max(outputs, 1)
            
            # 遍历批次中的每个样本
            for i in range(len(labels_batch)):
                true_label = labels_batch[i].item()
                pred_label = predicted[i].item()
                confidence = probs[i, pred_label].item()
                
                # 检查是否是我们想要的错误类型
                if true_label == true_class and pred_label == pred_class:
                    error_samples.append({
                        'sample_idx': sample_idx,
                        'image': images[i].cpu(),  # 保存原始tensor用于显示
                        'confidence': confidence,
                        'true_class': true_label,
                        'pred_class': pred_label,
                        'true_name': class_names[true_label],
                        'pred_name': class_names[pred_label]
                    })
                
                sample_idx += 1
    
    # 按置信度降序排序
    error_samples.sort(key=lambda x: x['confidence'], reverse=True)
    
    # 返回top_k个
    return error_samples[:top_k]


def denormalize_image(tensor, mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]):
    """
    反归一化图片tensor用于显示
    
    Args:
        tensor: 归一化后的tensor [C, H, W]
        mean: 归一化均值
        std: 归一化标准差
    
    Returns:
        numpy array [H, W, C] 值在[0, 1]范围
    """
    tensor = tensor.clone()
    for t, m, s in zip(tensor, mean, std):
        t.mul_(s).add_(m)
    tensor = torch.clamp(tensor, 0, 1)
    return tensor.permute(1, 2, 0).numpy()


def generate_error_report(model, test_loader, device, confused_pairs, class_names, 
                         output_path, top_k_per_pair=10):
    """
    生成错误分析报告（HTML格式）
    
    Args:
        model: 模型
        test_loader: 测试数据加载器
        device: 设备
        confused_pairs: 混淆的类别对列表
        class_names: 类别名称列表
        output_path: 输出HTML文件路径
        top_k_per_pair: 每对类别显示的错误样本数
    """
    html_content = []
    html_content.append("""
<!DOCTYPE html>
<html lang="zh-CN">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>错误分析报告 - 宠物分类模型</title>
    <style>
        body {
            font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
            margin: 0;
            padding: 20px;
            background-color: #f5f5f5;
        }
        .container {
            max-width: 1400px;
            margin: 0 auto;
            background-color: white;
            padding: 30px;
            border-radius: 10px;
            box-shadow: 0 2px 10px rgba(0,0,0,0.1);
        }
        h1 {
            color: #333;
            border-bottom: 3px solid #4CAF50;
            padding-bottom: 10px;
        }
        h2 {
            color: #555;
            margin-top: 30px;
            padding: 10px;
            background-color: #f0f0f0;
            border-left: 4px solid #4CAF50;
        }
        .pair-info {
            background-color: #fff3cd;
            padding: 15px;
            border-radius: 5px;
            margin: 20px 0;
            border-left: 4px solid #ffc107;
        }
        .pair-info strong {
            color: #856404;
        }
        .error-grid {
            display: grid;
            grid-template-columns: repeat(auto-fill, minmax(200px, 1fr));
            gap: 20px;
            margin: 20px 0;
        }
        .error-item {
            border: 1px solid #ddd;
            border-radius: 5px;
            padding: 10px;
            text-align: center;
            background-color: #fafafa;
        }
        .error-item img {
            width: 100%;
            height: auto;
            border-radius: 5px;
            margin-bottom: 10px;
        }
        .error-item .confidence {
            font-weight: bold;
            color: #d32f2f;
            margin-top: 5px;
        }
        .error-item .label {
            font-size: 0.9em;
            color: #666;
            margin-top: 5px;
        }
        .summary {
            background-color: #e3f2fd;
            padding: 20px;
            border-radius: 5px;
            margin: 20px 0;
        }
        .summary h3 {
            margin-top: 0;
            color: #1976d2;
        }
    </style>
</head>
<body>
    <div class="container">
        <h1>🐾 宠物分类模型 - 错误分析报告</h1>
        <div class="summary">
            <h3>报告摘要</h3>
            <p>本报告展示了模型最容易混淆的 <strong>5对类别</strong>，以及每对类别中置信度最高的错误样本。</p>
            <p>错误样本按置信度从高到低排序，帮助识别模型最"自信"但预测错误的案例。</p>
        </div>
    """)
    
    # 为每对混淆类别生成内容
    for pair_idx, pair in enumerate(confused_pairs, 1):
        print(f"\n处理混淆对 {pair_idx}/5: {pair['true_name']} -> {pair['pred_name']} (错误次数: {pair['count']})")
        
        # 获取错误样本
        error_samples = get_error_samples(
            model, test_loader, device,
            pair['true_class'], pair['pred_class'],
            class_names, top_k=top_k_per_pair
        )
        
        if not error_samples:
            continue
        
        html_content.append(f"""
        <h2>混淆对 #{pair_idx}: {pair['true_name']} → {pair['pred_name']}</h2>
        <div class="pair-info">
            <strong>真实类别:</strong> {pair['true_name']}<br>
            <strong>预测类别:</strong> {pair['pred_name']}<br>
            <strong>错误次数:</strong> {pair['count']}<br>
            <strong>显示样本数:</strong> {len(error_samples)}
        </div>
        <div class="error-grid">
        """)
        
        # 保存图片并生成HTML
        pair_dir = os.path.join(os.path.dirname(output_path), 'error_images', f'pair_{pair_idx}')
        os.makedirs(pair_dir, exist_ok=True)
        
        for sample_idx, sample in enumerate(error_samples):
            # 反归一化图片
            img_array = denormalize_image(sample['image'])
            img_array = (img_array * 255).astype(np.uint8)
            
            # 保存图片
            img_filename = f'sample_{sample_idx+1}.png'
            img_path = os.path.join(pair_dir, img_filename)
            Image.fromarray(img_array).save(img_path)
            
            # 生成相对路径用于HTML
            rel_img_path = os.path.relpath(img_path, os.path.dirname(output_path))
            
            html_content.append(f"""
            <div class="error-item">
                <img src="{rel_img_path}" alt="Error Sample {sample_idx+1}">
                <div class="confidence">置信度: {sample['confidence']:.4f}</div>
                <div class="label">真实: {sample['true_name']}</div>
                <div class="label">预测: {sample['pred_name']}</div>
            </div>
            """)
        
        html_content.append("</div>")
    
    html_content.append("""
    </div>
</body>
</html>
    """)
    
    # 保存HTML文件
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with open(output_path, 'w', encoding='utf-8') as f:
        f.write(''.join(html_content))
    
    print(f"\n错误分析报告已保存至: {output_path}")


def analyze_errors(args):
    """
    主错误分析函数
    
    Args:
        args: 命令行参数
    """
    # 如果output_path是相对路径，转换为项目根目录下的绝对路径
    if args.output_path and not os.path.isabs(args.output_path):
        project_root = get_project_root()
        if args.output_path.startswith('reports/'):
            args.output_path = str(project_root / args.output_path)
        else:
            args.output_path = str(project_root / 'reports' / os.path.basename(args.output_path))
    
    # 设备选择
    if torch.cuda.is_available():
        device = torch.device('cuda')
    elif torch.backends.mps.is_available():
        device = torch.device('mps')
    else:
        device = torch.device('cpu')
    
    print(f"使用设备: {device}")
    print(f"报告输出路径: {args.output_path}")
    
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
    
    # 评估模型获取混淆矩阵
    print("\n评估模型以获取混淆矩阵...")
    results = evaluate_model(model, test_loader, device, class_names)
    confusion_matrix = results['confusion_matrix']
    
    # 找出最容易混淆的5对类别
    print("\n分析混淆类别对...")
    confused_pairs = find_confused_pairs(confusion_matrix, class_names, top_k=5)
    
    print("\n最容易混淆的5对类别:")
    for i, pair in enumerate(confused_pairs, 1):
        print(f"  {i}. {pair['true_name']} -> {pair['pred_name']} (错误次数: {pair['count']})")
    
    # 生成错误分析报告
    print("\n生成错误分析报告...")
    generate_error_report(
        model, test_loader, device,
        confused_pairs, class_names,
        args.output_path,
        top_k_per_pair=args.top_k_per_pair
    )
    
    print("\n错误分析完成！")


def main():
    parser = argparse.ArgumentParser(description='错误分析：找出混淆的类别对')
    parser.add_argument('--model-path', type=str, required=True, help='模型权重路径')
    parser.add_argument('--data-dir', type=str, default=None, help='数据目录（默认：项目根目录下的data/）')
    parser.add_argument('--output-path', type=str, default=None, help='输出HTML报告路径（默认：reports/error_report.html）')
    parser.add_argument('--batch-size', type=int, default=32, help='批次大小')
    parser.add_argument('--num-classes', type=int, default=37, help='类别数')
    parser.add_argument('--num-workers', type=int, default=4, help='数据加载worker数量')
    parser.add_argument('--use-official-test', action='store_true', help='使用官方测试集')
    parser.add_argument('--top-k-per-pair', type=int, default=10, help='每对类别显示的错误样本数')
    
    args = parser.parse_args()
    # 设置默认输出路径
    if args.output_path is None:
        project_root = get_project_root()
        args.output_path = str(project_root / 'reports' / 'error_report.html')
    analyze_errors(args)


if __name__ == '__main__':
    main()
