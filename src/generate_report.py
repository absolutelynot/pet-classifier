#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
测试报告生成脚本
在官方测试集上运行评估，生成HTML格式的测试准确率报告
"""

import os
import torch
import numpy as np
import argparse
import sys
from datetime import datetime

# 添加src目录到路径
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from model import load_model
from data_loader import get_test_dataset
from evaluate import evaluate_model, plot_confusion_matrix


def generate_html_report(results, output_path, confusion_matrix_path=None):
    """
    生成HTML格式的测试报告
    
    Args:
        results: 评估结果字典
        output_path: 输出HTML文件路径
        confusion_matrix_path: 混淆矩阵图片路径（可选）
    """
    accuracy = results['accuracy']
    confusion_mat = results['confusion_matrix']
    class_names = results['class_names']
    report = results['classification_report']
    
    # 计算各类别准确率
    class_accuracies = []
    for i, class_name in enumerate(class_names):
        if i in report:
            precision = report[i]['precision']
            recall = report[i]['recall']
            f1 = report[i]['f1-score']
            support = report[i]['support']
            class_accuracies.append({
                'name': class_name,
                'precision': precision,
                'recall': recall,
                'f1': f1,
                'support': support
            })
        else:
            # 如果类别没有出现在测试集中
            class_accuracies.append({
                'name': class_name,
                'precision': 0.0,
                'recall': 0.0,
                'f1': 0.0,
                'support': 0
            })
    
    # 按准确率排序
    class_accuracies.sort(key=lambda x: x['precision'], reverse=True)
    
    html_content = []
    html_content.append("""
<!DOCTYPE html>
<html lang="zh-CN">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>测试准确率报告 - 宠物分类模型</title>
    <style>
        * {
            margin: 0;
            padding: 0;
            box-sizing: border-box;
        }
        body {
            font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            padding: 20px;
            min-height: 100vh;
        }
        .container {
            max-width: 1400px;
            margin: 0 auto;
            background-color: white;
            padding: 40px;
            border-radius: 15px;
            box-shadow: 0 10px 40px rgba(0,0,0,0.2);
        }
        h1 {
            color: #333;
            text-align: center;
            margin-bottom: 10px;
            font-size: 2.5em;
        }
        .subtitle {
            text-align: center;
            color: #666;
            margin-bottom: 30px;
            font-size: 1.1em;
        }
        .summary-card {
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            color: white;
            padding: 30px;
            border-radius: 10px;
            text-align: center;
            margin-bottom: 30px;
        }
        .summary-card h2 {
            font-size: 3em;
            margin-bottom: 10px;
        }
        .summary-card p {
            font-size: 1.2em;
            opacity: 0.9;
        }
        .section {
            margin: 30px 0;
        }
        .section h2 {
            color: #555;
            margin-bottom: 20px;
            padding-bottom: 10px;
            border-bottom: 3px solid #667eea;
        }
        .confusion-matrix {
            text-align: center;
            margin: 20px 0;
        }
        .confusion-matrix img {
            max-width: 100%;
            height: auto;
            border-radius: 10px;
            box-shadow: 0 4px 15px rgba(0,0,0,0.1);
        }
        table {
            width: 100%;
            border-collapse: collapse;
            margin: 20px 0;
            box-shadow: 0 2px 10px rgba(0,0,0,0.1);
        }
        thead {
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            color: white;
        }
        th, td {
            padding: 12px;
            text-align: left;
            border-bottom: 1px solid #ddd;
        }
        th {
            font-weight: 600;
        }
        tbody tr:hover {
            background-color: #f5f5f5;
        }
        .metric-badge {
            display: inline-block;
            padding: 5px 10px;
            border-radius: 5px;
            font-weight: bold;
            font-size: 0.9em;
        }
        .metric-excellent {
            background-color: #4CAF50;
            color: white;
        }
        .metric-good {
            background-color: #8BC34A;
            color: white;
        }
        .metric-fair {
            background-color: #FFC107;
            color: #333;
        }
        .metric-poor {
            background-color: #FF9800;
            color: white;
        }
        .metric-bad {
            background-color: #F44336;
            color: white;
        }
        .stats-grid {
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(200px, 1fr));
            gap: 20px;
            margin: 20px 0;
        }
        .stat-card {
            background-color: #f8f9fa;
            padding: 20px;
            border-radius: 10px;
            text-align: center;
            border-left: 4px solid #667eea;
        }
        .stat-card h3 {
            color: #667eea;
            margin-bottom: 10px;
        }
        .stat-card .value {
            font-size: 2em;
            font-weight: bold;
            color: #333;
        }
        .footer {
            text-align: center;
            margin-top: 40px;
            padding-top: 20px;
            border-top: 1px solid #ddd;
            color: #666;
        }
    </style>
</head>
<body>
    <div class="container">
        <h1>🐾 宠物分类模型测试报告</h1>
        <p class="subtitle">Oxford-IIIT Pet Dataset - 37类分类</p>
        
        <div class="summary-card">
            <h2>{:.2f}%</h2>
            <p>总体测试准确率</p>
        </div>
    """.format(accuracy))
    
    # 统计信息
    total_samples = len(results['true_labels'])
    correct_samples = np.sum(results['predictions'] == results['true_labels'])
    
    html_content.append("""
        <div class="stats-grid">
            <div class="stat-card">
                <h3>总样本数</h3>
                <div class="value">{}</div>
            </div>
            <div class="stat-card">
                <h3>正确预测</h3>
                <div class="value">{}</div>
            </div>
            <div class="stat-card">
                <h3>错误预测</h3>
                <div class="value">{}</div>
            </div>
            <div class="stat-card">
                <h3>类别数</h3>
                <div class="value">{}</div>
            </div>
        </div>
    """.format(total_samples, correct_samples, total_samples - correct_samples, len(class_names)))
    
    # 混淆矩阵
    if confusion_matrix_path and os.path.exists(confusion_matrix_path):
        rel_path = os.path.relpath(confusion_matrix_path, os.path.dirname(output_path))
        html_content.append(f"""
        <div class="section">
            <h2>混淆矩阵</h2>
            <div class="confusion-matrix">
                <img src="{rel_path}" alt="Confusion Matrix">
            </div>
        </div>
        """)
    
    # 各类别详细指标
    html_content.append("""
        <div class="section">
            <h2>各类别详细指标</h2>
            <table>
                <thead>
                    <tr>
                        <th>排名</th>
                        <th>类别名称</th>
                        <th>精确率 (Precision)</th>
                        <th>召回率 (Recall)</th>
                        <th>F1分数</th>
                        <th>样本数</th>
                    </tr>
                </thead>
                <tbody>
    """)
    
    for idx, class_info in enumerate(class_accuracies, 1):
        precision = class_info['precision']
        recall = class_info['recall']
        f1 = class_info['f1']
        support = class_info['support']
        
        # 根据准确率选择badge样式
        if precision >= 0.9:
            badge_class = 'metric-excellent'
        elif precision >= 0.7:
            badge_class = 'metric-good'
        elif precision >= 0.5:
            badge_class = 'metric-fair'
        elif precision >= 0.3:
            badge_class = 'metric-poor'
        else:
            badge_class = 'metric-bad'
        
        html_content.append(f"""
                    <tr>
                        <td>{idx}</td>
                        <td><strong>{class_info['name']}</strong></td>
                        <td><span class="metric-badge {badge_class}">{precision:.2%}</span></td>
                        <td><span class="metric-badge {badge_class}">{recall:.2%}</span></td>
                        <td><span class="metric-badge {badge_class}">{f1:.2%}</span></td>
                        <td>{support}</td>
                    </tr>
        """)
    
    html_content.append("""
                </tbody>
            </table>
        </div>
    """)
    
    # 页脚
    html_content.append(f"""
        <div class="footer">
            <p>报告生成时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}</p>
            <p>模型: EfficientNet-B2 (仅微调分类头)</p>
        </div>
    </div>
</body>
</html>
    """)
    
    # 保存HTML文件
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with open(output_path, 'w', encoding='utf-8') as f:
        f.write(''.join(html_content))
    
    print(f"测试报告已保存至: {output_path}")


def generate_report(args):
    """
    主报告生成函数
    
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
    
    # 加载官方测试集
    print("加载官方测试集...")
    test_loader, num_classes, class_names = get_test_dataset(
        data_dir=args.data_dir,
        batch_size=args.batch_size,
        num_workers=args.num_workers
    )
    
    # 评估模型
    print("评估模型...")
    results = evaluate_model(model, test_loader, device, class_names)
    
    # 打印总体准确率
    print(f"\n总体测试准确率: {results['accuracy']:.2f}%")
    
    # 生成混淆矩阵
    confusion_matrix_path = None
    if args.save_confusion_matrix:
        confusion_matrix_path = os.path.join(args.output_dir, 'confusion_matrix.png')
        os.makedirs(args.output_dir, exist_ok=True)
        plot_confusion_matrix(
            results['confusion_matrix'],
            results['class_names'],
            confusion_matrix_path
        )
    
    # 生成HTML报告
    print("生成HTML报告...")
    report_path = os.path.join(args.output_dir, 'test_report.html')
    generate_html_report(results, report_path, confusion_matrix_path)
    
    print("\n报告生成完成！")


def main():
    parser = argparse.ArgumentParser(description='生成测试准确率报告')
    parser.add_argument('--model-path', type=str, required=True, help='模型权重路径')
    parser.add_argument('--data-dir', type=str, default=None, help='数据目录（默认：项目根目录下的data/）')
    parser.add_argument('--output-dir', type=str, default='reports', help='输出目录')
    parser.add_argument('--batch-size', type=int, default=32, help='批次大小')
    parser.add_argument('--num-classes', type=int, default=37, help='类别数')
    parser.add_argument('--num-workers', type=int, default=4, help='数据加载worker数量')
    parser.add_argument('--save-confusion-matrix', action='store_true', help='保存混淆矩阵')
    
    args = parser.parse_args()
    generate_report(args)


if __name__ == '__main__':
    main()
