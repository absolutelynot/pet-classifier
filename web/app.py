#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
FastAPI Web应用
提供图片上传、分类预测、历史记录查询等功能
"""

import os
import sys
import logging
import uuid
from pathlib import Path
from datetime import datetime

import torch
from torchvision import transforms
from PIL import Image
import uvicorn
from fastapi import FastAPI, File, UploadFile, HTTPException, Form, Request
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates

# 添加项目根目录到路径
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))
sys.path.insert(0, str(project_root / 'src'))


def get_project_root():
    """获取项目根目录（pet-classifier目录）"""
    current_file = Path(__file__).resolve()
    # 从当前文件向上查找，直到找到包含 .git 或 requirements.txt 的目录
    for parent in current_file.parents:
        if (parent / 'requirements.txt').exists() or (parent / '.git').exists():
            return parent
    # 如果找不到，返回当前文件的上两级目录（假设在 web/ 下）
    return current_file.parent.parent

from src.model import load_model
from web.database import get_database

# 配置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(str(project_root / 'web' / 'app.log')),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

# 创建FastAPI应用
app = FastAPI(title="宠物分类API", version="1.0.0")

# 配置模板和静态文件
templates_dir = project_root / "web" / "templates"
static_dir = project_root / "web" / "static"
templates = Jinja2Templates(directory=str(templates_dir))
app.mount("/static", StaticFiles(directory=str(static_dir)), name="static")
# 挂载uploads目录
uploads_dir = project_root / "web" / "uploads"
uploads_dir.mkdir(parents=True, exist_ok=True)
app.mount("/uploads", StaticFiles(directory=str(uploads_dir)), name="uploads")

# 全局变量
model = None
class_names = None
device = None
db = None
upload_dir = project_root / "web" / "uploads"
upload_dir.mkdir(parents=True, exist_ok=True)

# 数据预处理
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], 
                        std=[0.229, 0.224, 0.225])
])


def load_classifier_model(model_path='models/best_model.pth', num_classes=37):
    """
    加载分类模型
    
    Args:
        model_path: 模型权重路径
        num_classes: 类别数
    """
    global model, device, class_names
    
    # 设备选择
    if torch.cuda.is_available():
        device = torch.device('cuda')
        logger.info("使用CUDA")
    elif torch.backends.mps.is_available():
        device = torch.device('mps')
        logger.info("使用MPS (Apple Silicon)")
    else:
        device = torch.device('cpu')
        logger.info("使用CPU")
    
    # 加载模型
    logger.info(f"加载模型: {model_path}")
    model = load_model(model_path, num_classes=num_classes, device=device)
    logger.info("模型加载完成")
    
    # 加载类别名称（从数据加载器获取）
    try:
        from src.data_loader import get_data_loaders
        # 使用None让get_data_loaders自动检测项目根目录下的data/
        _, _, _, _, class_names = get_data_loaders(
            data_dir=None,  # 自动使用项目根目录下的data/
            batch_size=1,
            num_workers=0
        )
        logger.info(f"类别名称加载完成，共{len(class_names)}个类别")
    except Exception as e:
        logger.warning(f"无法加载类别名称: {e}")
        class_names = [f'Class {i}' for i in range(num_classes)]


def predict_image(image_path: str):
    """
    对图片进行预测
    
    Args:
        image_path: 图片路径
    
    Returns:
        (predicted_class, confidence, class_index)
    """
    global model, device, class_names
    
    if model is None:
        raise HTTPException(status_code=500, detail="模型未加载")
    
    try:
        # 加载和预处理图片
        image = Image.open(image_path).convert('RGB')
        image_tensor = transform(image).unsqueeze(0).to(device)
        
        # 预测
        model.eval()
        with torch.no_grad():
            outputs = model(image_tensor)
            probs = torch.softmax(outputs, dim=1)
            confidence, predicted = torch.max(probs, 1)
        
        class_index = predicted.item()
        confidence_value = confidence.item()
        predicted_class = class_names[class_index] if class_names else f'Class {class_index}'
        
        logger.info(f"预测结果: {predicted_class} (置信度: {confidence_value:.4f})")
        
        return predicted_class, confidence_value, class_index
    
    except Exception as e:
        logger.error(f"预测失败: {e}")
        raise HTTPException(status_code=500, detail=f"预测失败: {str(e)}")


@app.on_event("startup")
async def startup_event():
    """应用启动时加载模型"""
    global db
    db = get_database()
    logger.info("数据库初始化完成")
    
    # 尝试加载模型
    # 如果MODEL_PATH环境变量未设置，使用项目根目录下的models/best_model.pth
    model_path = os.getenv('MODEL_PATH')
    if model_path is None:
        project_root = get_project_root()
        model_path = str(project_root / 'models' / 'best_model.pth')
    
    if os.path.exists(model_path):
        logger.info(f"找到模型文件: {model_path}")
        load_classifier_model(model_path)
    else:
        logger.warning(f"模型文件不存在: {model_path}，请先训练模型")


@app.get("/", response_class=HTMLResponse)
async def read_root(request: Request):
    """主页"""
    return templates.TemplateResponse("index.html", {"request": request})


@app.post("/upload")
async def upload_file(file: UploadFile = File(...)):
    """
    上传图片并进行分类
    
    Args:
        file: 上传的图片文件
    
    Returns:
        JSON响应，包含预测结果
    """
    logger.info(f"收到上传请求: {file.filename}")
    
    # 验证文件类型
    if not file.content_type.startswith('image/'):
        raise HTTPException(status_code=400, detail="文件必须是图片格式")
    
    # 保存上传的文件
    file_ext = Path(file.filename).suffix
    unique_filename = f"{uuid.uuid4()}{file_ext}"
    file_path = upload_dir / unique_filename
    
    try:
        with open(file_path, "wb") as buffer:
            content = await file.read()
            buffer.write(content)
        
        logger.info(f"文件已保存: {file_path}")
        
        # 进行预测
        predicted_class, confidence, class_index = predict_image(str(file_path))
        
        # 保存到数据库
        prediction_id = db.add_prediction(
            image_path=str(file_path),
            predicted_class=predicted_class,
            confidence=confidence,
            class_index=class_index
        )
        
        logger.info(f"预测结果已保存到数据库，ID: {prediction_id}")
        
        # 生成图片URL（相对于static目录）
        image_url = f"/uploads/{unique_filename}"
        
        return JSONResponse({
            "success": True,
            "prediction_id": prediction_id,
            "predicted_class": predicted_class,
            "confidence": float(confidence),
            "class_index": class_index,
            "image_url": image_url
        })
    
    except Exception as e:
        logger.error(f"处理上传文件时出错: {e}")
        # 如果出错，删除已保存的文件
        if file_path.exists():
            file_path.unlink()
        raise HTTPException(status_code=500, detail=f"处理文件时出错: {str(e)}")


@app.get("/predict/{prediction_id}")
async def get_prediction(prediction_id: int):
    """
    根据ID获取预测结果
    
    Args:
        prediction_id: 预测记录ID
    
    Returns:
        JSON响应，包含预测结果
    """
    prediction = db.get_prediction(prediction_id)
    
    if prediction is None:
        raise HTTPException(status_code=404, detail="预测记录不存在")
    
    return JSONResponse(prediction)


@app.get("/history")
async def get_history(limit: int = 20, offset: int = 0):
    """
    获取预测历史记录
    
    Args:
        limit: 返回记录数限制
        offset: 偏移量
    
    Returns:
        JSON响应，包含历史记录列表
    """
    predictions = db.get_all_predictions(limit=limit, offset=offset)
    
    # 转换图片路径为URL
    for pred in predictions:
        if pred['image_path']:
            filename = Path(pred['image_path']).name
            pred['image_url'] = f"/uploads/{filename}"
    
    return JSONResponse({
        "success": True,
        "count": len(predictions),
        "predictions": predictions
    })


@app.get("/statistics")
async def get_statistics():
    """
    获取统计信息
    
    Returns:
        JSON响应，包含统计信息
    """
    stats = db.get_statistics()
    return JSONResponse(stats)


@app.get("/health")
async def health_check():
    """健康检查"""
    return JSONResponse({
        "status": "healthy",
        "model_loaded": model is not None,
        "database_connected": db is not None
    })


if __name__ == "__main__":
    # 运行应用
    uvicorn.run(
        "app:app",
        host="0.0.0.0",
        port=8000,
        reload=True,
        log_level="info"
    )
