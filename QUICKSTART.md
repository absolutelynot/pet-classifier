# 快速开始指南

## 1. 安装依赖

```bash
pip install -r requirements.txt
```

## 2. 训练模型（必需）

在训练模型之前，Web应用无法正常工作。

```bash
cd src
python train.py --epochs 20
```

训练完成后，模型会保存在 `models/best_model.pth`。

## 3. 生成报告（可选）

### 错误分析报告

```bash
cd src
python error_analysis.py --model-path ../models/best_model.pth
```

### 测试准确率报告

```bash
cd src
python generate_report.py --model-path ../models/best_model.pth --use-official-test --save-confusion-matrix
```

## 4. 启动Web应用

```bash
# 方法1: 使用便捷脚本
bash run_web.sh

# 方法2: 直接运行
cd web
python app.py

# 方法3: 使用uvicorn
cd web
uvicorn app:app --host 0.0.0.0 --port 8000 --reload
```

然后在浏览器中访问：`http://localhost:8000`

## 5. 使用Web界面

1. 打开浏览器访问 `http://localhost:8000`
2. 点击上传区域或拖拽图片
3. 点击"开始分类"按钮
4. 查看预测结果和历史记录

## 注意事项

- 首次运行需要下载数据集（约800MB），请确保网络连接正常
- 训练过程可能需要30-60分钟（取决于硬件配置）
- Web应用需要模型文件存在才能正常工作
- 上传的图片会保存在 `web/uploads/` 目录
- 预测结果会保存在 `database/predictions.db` 数据库中
