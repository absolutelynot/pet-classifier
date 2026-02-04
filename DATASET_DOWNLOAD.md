# Oxford-IIIT Pet 数据集手动下载指南

由于网络下载速度较慢，您可以通过以下方式手动下载数据集：

## 官方下载链接

Oxford-IIIT Pet数据集包含两个文件：

1. **图片文件** (images.tar.gz, 约 733MB)
   ```
   https://www.robots.ox.ac.uk/~vgg/data/pets/data/images.tar.gz
   ```

2. **标注文件** (annotations.tar.gz, 约 3.5MB)
   ```
   https://www.robots.ox.ac.uk/~vgg/data/pets/data/annotations.tar.gz
   ```

## 下载步骤

### 方法1：使用浏览器直接下载

1. 打开浏览器，访问以下链接：
   - 图片文件：https://www.robots.ox.ac.uk/~vgg/data/pets/data/images.tar.gz
   - 标注文件：https://www.robots.ox.ac.uk/~vgg/data/pets/data/annotations.tar.gz

2. 下载完成后，将文件放到以下目录：
   ```
   data/oxford-iiit-pet/
   ```

3. 解压文件：
   ```bash
   cd data/oxford-iiit-pet/
   tar -xzf images.tar.gz
   tar -xzf annotations.tar.gz
   ```

4. 最终目录结构应该是：
   ```
   data/oxford-iiit-pet/
   ├── images/
   │   ├── Abyssinian_001.jpg
   │   ├── Abyssinian_002.jpg
   │   └── ...
   └── annotations/
       ├── trimaps/
       └── xmls/
   ```

### 方法2：使用下载工具（推荐）

使用支持断点续传的下载工具，如：
- **aria2** (命令行)
- **IDM** (Windows)
- **Folx** (macOS)
- **uGet** (Linux)

使用aria2下载示例：
```bash
# 安装aria2 (macOS)
brew install aria2

# 下载文件
cd data/oxford-iiit-pet/
aria2c -x 16 -s 16 https://www.robots.ox.ac.uk/~vgg/data/pets/data/images.tar.gz
aria2c -x 16 -s 16 https://www.robots.ox.ac.uk/~vgg/data/pets/data/annotations.tar.gz
```

### 方法3：使用镜像源（如果可用）

如果官方链接速度慢，可以尝试：
- 使用VPN或代理
- 查找国内镜像源
- 使用云盘分享（如果有）

## 验证下载

下载并解压后，运行以下命令验证：

```bash
cd src
python -c "from data_loader import get_data_loaders; get_data_loaders(data_dir='../data', batch_size=1, num_workers=0)"
```

如果看到类似以下输出，说明数据集已正确下载：
```
数据集类别数: 37
数据集总样本数: 7349
训练集样本数: 5879
验证集样本数: 1102
测试集样本数: 368
```

## 注意事项

1. **文件完整性**：确保下载的文件完整，如果下载中断，请重新下载
2. **磁盘空间**：解压后需要约 1.5GB 的磁盘空间
3. **文件权限**：确保有读写权限
4. **路径问题**：确保文件放在正确的目录 `data/oxford-iiit-pet/` 下

## 如果下载仍然很慢

可以考虑：
1. 使用下载工具的多线程下载功能
2. 在网速好的时候/地点下载
3. 使用云服务器下载后传输到本地
4. 联系项目维护者获取网盘链接
