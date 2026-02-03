#!/bin/bash
# 启动Web应用的便捷脚本

cd "$(dirname "$0")"
cd web
python app.py
