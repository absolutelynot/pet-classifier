#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
数据库模块
使用SQLite存储上传的图片和分类结果
"""

import os
import sqlite3
from datetime import datetime
from typing import List, Dict, Optional
from pathlib import Path


def get_project_root():
    """获取项目根目录（pet-classifier目录）"""
    current_file = Path(__file__).resolve()
    # 从当前文件向上查找，直到找到包含 .git 或 requirements.txt 的目录
    for parent in current_file.parents:
        if (parent / 'requirements.txt').exists() or (parent / '.git').exists():
            return parent
    # 如果找不到，返回当前文件的上两级目录（假设在 web/ 下）
    return current_file.parent.parent


class PredictionDatabase:
    """预测结果数据库"""
    
    def __init__(self, db_path=None):
        """
        初始化数据库
        
        Args:
            db_path: 数据库文件路径（如果为None，则使用项目根目录下的database/predictions.db）
        """
        # 如果未指定db_path，使用项目根目录下的database/
        if db_path is None:
            project_root = get_project_root()
            db_path = str(project_root / 'database' / 'predictions.db')
        
        self.db_path = db_path
        os.makedirs(os.path.dirname(db_path), exist_ok=True)
        self._init_database()
    
    def _init_database(self):
        """初始化数据库表结构"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        # 创建预测结果表
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS predictions (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                image_path TEXT NOT NULL,
                predicted_class TEXT NOT NULL,
                confidence REAL NOT NULL,
                upload_time TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                class_index INTEGER
            )
        ''')
        
        # 创建索引以提高查询性能
        cursor.execute('''
            CREATE INDEX IF NOT EXISTS idx_upload_time 
            ON predictions(upload_time DESC)
        ''')
        
        cursor.execute('''
            CREATE INDEX IF NOT EXISTS idx_predicted_class 
            ON predictions(predicted_class)
        ''')
        
        conn.commit()
        conn.close()
    
    def add_prediction(self, image_path: str, predicted_class: str, 
                      confidence: float, class_index: int = None) -> int:
        """
        添加预测结果
        
        Args:
            image_path: 图片路径
            predicted_class: 预测的类别名称
            confidence: 置信度
            class_index: 类别索引（可选）
        
        Returns:
            插入记录的ID
        """
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        cursor.execute('''
            INSERT INTO predictions 
            (image_path, predicted_class, confidence, class_index, upload_time)
            VALUES (?, ?, ?, ?, ?)
        ''', (image_path, predicted_class, confidence, class_index, datetime.now()))
        
        record_id = cursor.lastrowid
        conn.commit()
        conn.close()
        
        return record_id
    
    def get_prediction(self, prediction_id: int) -> Optional[Dict]:
        """
        根据ID获取预测结果
        
        Args:
            prediction_id: 预测记录ID
        
        Returns:
            预测结果字典，如果不存在返回None
        """
        conn = sqlite3.connect(self.db_path)
        conn.row_factory = sqlite3.Row
        cursor = conn.cursor()
        
        cursor.execute('''
            SELECT * FROM predictions WHERE id = ?
        ''', (prediction_id,))
        
        row = cursor.fetchone()
        conn.close()
        
        if row:
            return dict(row)
        return None
    
    def get_all_predictions(self, limit: int = 100, offset: int = 0) -> List[Dict]:
        """
        获取所有预测结果（按时间倒序）
        
        Args:
            limit: 返回记录数限制
            offset: 偏移量
        
        Returns:
            预测结果列表
        """
        conn = sqlite3.connect(self.db_path)
        conn.row_factory = sqlite3.Row
        cursor = conn.cursor()
        
        cursor.execute('''
            SELECT * FROM predictions 
            ORDER BY upload_time DESC 
            LIMIT ? OFFSET ?
        ''', (limit, offset))
        
        rows = cursor.fetchall()
        conn.close()
        
        return [dict(row) for row in rows]
    
    def get_predictions_by_class(self, predicted_class: str, 
                                 limit: int = 100) -> List[Dict]:
        """
        根据类别获取预测结果
        
        Args:
            predicted_class: 类别名称
            limit: 返回记录数限制
        
        Returns:
            预测结果列表
        """
        conn = sqlite3.connect(self.db_path)
        conn.row_factory = sqlite3.Row
        cursor = conn.cursor()
        
        cursor.execute('''
            SELECT * FROM predictions 
            WHERE predicted_class = ?
            ORDER BY upload_time DESC 
            LIMIT ?
        ''', (predicted_class, limit))
        
        rows = cursor.fetchall()
        conn.close()
        
        return [dict(row) for row in rows]
    
    def get_statistics(self) -> Dict:
        """
        获取数据库统计信息
        
        Returns:
            统计信息字典
        """
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        # 总记录数
        cursor.execute('SELECT COUNT(*) FROM predictions')
        total_count = cursor.fetchone()[0]
        
        # 平均置信度
        cursor.execute('SELECT AVG(confidence) FROM predictions')
        avg_confidence = cursor.fetchone()[0] or 0.0
        
        # 各类别统计
        cursor.execute('''
            SELECT predicted_class, COUNT(*) as count, AVG(confidence) as avg_conf
            FROM predictions
            GROUP BY predicted_class
            ORDER BY count DESC
        ''')
        
        class_stats = []
        for row in cursor.fetchall():
            class_stats.append({
                'class': row[0],
                'count': row[1],
                'avg_confidence': row[2]
            })
        
        conn.close()
        
        return {
            'total_predictions': total_count,
            'average_confidence': avg_confidence,
            'class_statistics': class_stats
        }
    
    def delete_prediction(self, prediction_id: int) -> bool:
        """
        删除预测记录
        
        Args:
            prediction_id: 预测记录ID
        
        Returns:
            是否删除成功
        """
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        cursor.execute('DELETE FROM predictions WHERE id = ?', (prediction_id,))
        deleted = cursor.rowcount > 0
        
        conn.commit()
        conn.close()
        
        return deleted


# 全局数据库实例
_db_instance = None


def get_database(db_path=None) -> PredictionDatabase:
    """
    获取数据库实例（单例模式）
    
    Args:
        db_path: 数据库文件路径（如果为None，则使用项目根目录下的database/predictions.db）
    
    Returns:
        数据库实例
    """
    global _db_instance
    if _db_instance is None:
        _db_instance = PredictionDatabase(db_path)
    return _db_instance


if __name__ == '__main__':
    # 测试数据库功能
    db = PredictionDatabase('database/test.db')
    
    # 添加测试数据
    test_id = db.add_prediction(
        image_path='test/image.jpg',
        predicted_class='Abyssinian',
        confidence=0.95,
        class_index=0
    )
    print(f"添加预测记录，ID: {test_id}")
    
    # 获取预测结果
    prediction = db.get_prediction(test_id)
    print(f"获取预测结果: {prediction}")
    
    # 获取统计信息
    stats = db.get_statistics()
    print(f"统计信息: {stats}")
    
    # 清理测试数据库
    os.remove('database/test.db')
    print("测试完成！")
