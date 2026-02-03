#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
模型定义模块
使用EfficientNet-B2作为backbone，只微调分类头
"""

import torch
import torch.nn as nn
from torchvision.models import efficientnet_b2, EfficientNet_B2_Weights


class PetClassifier(nn.Module):
    """
    宠物分类模型
    基于EfficientNet-B2，只微调分类头
    """
    
    def __init__(self, num_classes=37, freeze_backbone=True):
        """
        初始化模型
        
        Args:
            num_classes: 分类类别数（Oxford-IIIT Pet有37类）
            freeze_backbone: 是否冻结backbone参数
        """
        super(PetClassifier, self).__init__()
        
        # 加载预训练的EfficientNet-B2
        self.backbone = efficientnet_b2(weights=EfficientNet_B2_Weights.IMAGENET1K_V1)
        
        # 冻结backbone参数（只微调分类头）
        if freeze_backbone:
            for param in self.backbone.features.parameters():
                param.requires_grad = False
            for param in self.backbone.avgpool.parameters():
                param.requires_grad = False
        
        # 获取分类器的输入特征数
        in_features = self.backbone.classifier[1].in_features
        
        # 替换分类头为37类输出
        self.backbone.classifier = nn.Sequential(
            nn.Dropout(p=0.3, inplace=True),
            nn.Linear(in_features, num_classes)
        )
        
        self.num_classes = num_classes
    
    def forward(self, x):
        """
        前向传播
        
        Args:
            x: 输入图像张量 [batch_size, 3, 224, 224]
        
        Returns:
            分类logits [batch_size, num_classes]
        """
        return self.backbone(x)
    
    def get_trainable_parameters(self):
        """
        获取需要训练的参数（只返回分类头的参数）
        
        Returns:
            需要训练的参数列表
        """
        trainable_params = []
        for name, param in self.named_parameters():
            if param.requires_grad:
                trainable_params.append(param)
        return trainable_params
    
    def count_parameters(self):
        """
        统计模型参数数量
        
        Returns:
            total_params: 总参数数
            trainable_params: 可训练参数数
        """
        total_params = sum(p.numel() for p in self.parameters())
        trainable_params = sum(p.numel() for p in self.parameters() if p.requires_grad)
        return total_params, trainable_params


def create_model(num_classes=37, freeze_backbone=True, device='cpu'):
    """
    创建并初始化模型
    
    Args:
        num_classes: 分类类别数
        freeze_backbone: 是否冻结backbone
        device: 设备（'cpu', 'cuda', 'mps'）
    
    Returns:
        模型实例
    """
    model = PetClassifier(num_classes=num_classes, freeze_backbone=freeze_backbone)
    model = model.to(device)
    
    # 打印模型信息
    total_params, trainable_params = model.count_parameters()
    print(f"模型总参数数: {total_params:,}")
    print(f"可训练参数数: {trainable_params:,}")
    print(f"冻结参数数: {total_params - trainable_params:,}")
    
    return model


def load_model(model_path, num_classes=37, device='cpu'):
    """
    加载训练好的模型
    
    Args:
        model_path: 模型权重文件路径
        num_classes: 分类类别数
        device: 设备
    
    Returns:
        加载权重的模型
    """
    model = create_model(num_classes=num_classes, device=device)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()
    return model


if __name__ == '__main__':
    # 测试模型
    print("测试模型创建...")
    device = torch.device('mps' if torch.backends.mps.is_available() else 'cpu')
    print(f"使用设备: {device}")
    
    model = create_model(num_classes=37, device=device)
    
    # 测试前向传播
    dummy_input = torch.randn(1, 3, 224, 224).to(device)
    output = model(dummy_input)
    print(f"输入形状: {dummy_input.shape}")
    print(f"输出形状: {output.shape}")
    print("模型测试完成！")
