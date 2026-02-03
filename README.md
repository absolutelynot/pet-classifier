# 宠物分类模型微调与Web展示项目

基于Oxford-IIIT Pet数据集的37类宠物分类系统，使用EfficientNet-B2模型（仅微调分类头），包含完整的训练、评估、错误分析和Web展示功能。

## 项目概述

本项目实现了以下功能：

- ✅ 在Oxford-IIIT Pet数据集上进行37类分类
- ✅ 使用EfficientNet-B2作为backbone，只微调分类头
- ✅ 找出模型最容易混淆的5对类别及其top错误图片
- ✅ 生成HTML格式的测试准确率报告
- ✅ Web界面：用户可以上传宠物图片，得到分类结果
- ✅ 数据库存储：所有预测结果保存到SQLite数据库

## 技术栈

- **深度学习框架**: PyTorch 2.0+
- **模型**: EfficientNet-B2 (torchvision)
- **Web框架**: FastAPI
- **数据库**: SQLite
- **前端**: HTML + JavaScript + Bootstrap 5
- **可视化**: Matplotlib, Seaborn

## 项目结构

```
pet-classifier/
├── data/                    # 数据集目录
│   ├── oxford-iiit-pet/     # 下载的数据集
│   └── processed/           # 预处理后的数据
├── models/                  # 保存的模型
│   ├── best_model.pth      # 最佳模型权重
│   └── confusion_matrix.png # 混淆矩阵
├── src/                     # 源代码
│   ├── data_loader.py       # 数据加载和预处理
│   ├── model.py            # 模型定义
│   ├── train.py            # 训练脚本
│   ├── evaluate.py         # 评估脚本
│   ├── error_analysis.py   # 错误分析脚本
│   └── generate_report.py  # 生成HTML测试报告
├── web/                     # Web应用
│   ├── app.py              # FastAPI后端
│   ├── database.py         # 数据库操作
│   ├── templates/          # HTML模板
│   │   └── index.html      # 主页面
│   └── static/             # 静态文件
├── reports/                 # 生成的报告
│   ├── error_report.html   # 错误样例报告
│   └── test_report.html    # 测试准确率报告
├── database/                # 数据库文件
│   └── predictions.db      # SQLite数据库
├── requirements.txt         # Python依赖
└── README.md               # 项目说明
```

## 安装步骤

### 1. 克隆项目

```bash
cd /Users/wyy/Desktop/code/examine-202602
cd pet-classifier
```

### 2. 创建虚拟环境（推荐）

```bash
python3 -m venv venv
source venv/bin/activate  # macOS/Linux
# 或
venv\Scripts\activate  # Windows
```

### 3. 安装依赖

```bash
pip install -r requirements.txt
```

**注意**: 如果使用Apple Silicon (M1/M2)，PyTorch会自动使用MPS加速。如果使用CUDA，请根据[PyTorch官网](https://pytorch.org/)安装对应版本的PyTorch。

## 使用指南

### 1. 训练模型

训练模型（只微调分类头）：

```bash
cd src
python train.py --epochs 20 --batch-size 32 --learning-rate 0.001
```

参数说明：
- `--epochs`: 训练轮数（默认20）
- `--batch-size`: 批次大小（默认32）
- `--learning-rate`: 学习率（默认0.001）
- `--data-dir`: 数据目录（默认`../data`）
- `--model-dir`: 模型保存目录（默认`../models`）

训练完成后，最佳模型会保存在 `models/best_model.pth`。

### 2. 评估模型

在测试集上评估模型：

```bash
python evaluate.py --model-path ../models/best_model.pth --save-confusion-matrix
```

生成混淆矩阵和评估结果。

### 3. 错误分析

生成错误分析报告（找出最容易混淆的5对类别）：

```bash
python error_analysis.py --model-path ../models/best_model.pth --output-path ../reports/error_report.html
```

报告会保存在 `reports/error_report.html`，包含：
- 最容易混淆的5对类别
- 每对类别的top错误图片（按置信度排序）

### 4. 生成测试报告

在官方测试集上生成HTML格式的测试准确率报告：

```bash
python generate_report.py --model-path ../models/best_model.pth --use-official-test --save-confusion-matrix
```

报告会保存在 `reports/test_report.html`。

### 5. 启动Web应用

启动Web服务：

```bash
cd web
python app.py
```

或者使用uvicorn：

```bash
cd web
uvicorn app:app --host 0.0.0.0 --port 8000 --reload
```

然后在浏览器中访问：`http://localhost:8000`

**注意**: 首次运行需要先训练模型，确保 `models/best_model.pth` 存在。

## Web API接口

### 上传图片并预测

```bash
POST /upload
Content-Type: multipart/form-data

file: <图片文件>
```

响应：
```json
{
  "success": true,
  "prediction_id": 1,
  "predicted_class": "Abyssinian",
  "confidence": 0.95,
  "class_index": 0,
  "image_url": "/static/uploads/xxx.jpg"
}
```

### 获取预测历史

```bash
GET /history?limit=20&offset=0
```

### 获取统计信息

```bash
GET /statistics
```

### 健康检查

```bash
GET /health
```

## 交付物清单

1. ✅ **代码仓库及Git提交记录**
   - 完整的项目代码
   - Git提交历史

2. ✅ **错误样例报告**
   - `reports/error_report.html`
   - 包含最容易混淆的5对类别
   - 每对类别的top错误图片（按置信度排序）

3. ✅ **测试准确率报告**
   - `reports/test_report.html`
   - 在官方测试集上的准确率
   - 各类别详细指标

4. ✅ **Web应用**
   - 图片上传和分类功能
   - 预测结果存储到数据库
   - 历史记录查看

5. ✅ **录屏文件**（需要用户自行录制）
   - 包含操作演示
   - 后端调用日志

## 模型架构

- **Backbone**: EfficientNet-B2 (ImageNet预训练)
- **分类头**: 线性层 (1280 -> 37)
- **训练策略**: 冻结backbone，只训练分类头
- **优化器**: Adam (lr=0.001, weight_decay=1e-4)
- **学习率调度**: StepLR (step_size=7, gamma=0.1)

## 数据集信息

- **数据集**: Oxford-IIIT Pet
- **类别数**: 37
- **每类样本数**: 约200张
- **图片尺寸**: 224x224
- **数据增强**: 随机水平翻转、旋转、颜色抖动

## 常见问题

### Q: 训练需要多长时间？

A: 在Mac M1/M2上，使用MPS加速，训练20个epoch大约需要30-60分钟（取决于数据集大小和批次大小）。

### Q: 如何提高准确率？

A: 
- 增加训练轮数
- 调整学习率
- 使用更大的批次大小
- 尝试不同的数据增强策略
- 微调更多层（不仅限于分类头）

### Q: Web应用无法启动？

A: 
1. 确保已安装所有依赖：`pip install -r requirements.txt`
2. 确保模型文件存在：`models/best_model.pth`
3. 检查端口8000是否被占用

### Q: 如何查看训练日志？

A: 训练过程中的日志会输出到控制台。Web应用的日志保存在 `web/app.log`。

## 开发说明

### 代码组织

- `src/`: 模型训练和评估相关代码
- `web/`: Web应用相关代码
- `data/`: 数据集（自动下载）
- `models/`: 训练好的模型权重
- `reports/`: 生成的报告

### 扩展功能

可以轻松扩展以下功能：
- 支持批量上传
- 添加模型对比功能
- 实现用户认证
- 添加模型热更新
- 支持更多图片格式

## 许可证

本项目仅用于学习和研究目的。

## 联系方式

如有问题或建议，请提交Issue或Pull Request。

---

**注意**: 本项目遵循Vibe Coding开发理念，注重代码质量和可维护性。
