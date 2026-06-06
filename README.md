# 猫狗分类器 (Cat vs Dog Classifier)

基于 ResNet18 卷积神经网络实现的猫狗图像二分类项目，支持训练、验证和单张图片检测。

---

## 硬件配置

| 项目 | 型号 / 规格 |
|------|-------------|
| GPU | NVIDIA GeForce RTX 4070 Laptop GPU（8 GB 显存） |
| CPU | 13th Gen Intel(R) Core(TM) i7-13650HX (2.60 GHz)  |

## 软件及环境

| 软件 | 版本               |
|------|------------------|
| 操作系统 | Windows 11       |
| Python | 3.12.7           |
| CUDA | 13.0（PyTorch 内置） |
| GPU 驱动 | 592.27           |

## 依赖框架与库

| 库 | 版本 | 用途 |
|----|------|------|
| PyTorch | 2.12.0+cu130 | 深度学习框架，负责模型定义、训练与推理 |
| torchvision | 0.27.0+cu130 | 提供 ResNet 预定义模型与图像预处理工具 |
| Pillow | 10.4.0 | 图像读取与格式转换 |
| tqdm | 4.66.5 | 训练进度条显示 |
| tkinter | Python 内置 | 图形界面文件选择对话框 |

## 模型

采用 **ResNet18** 作为骨干网络，修改最后的全连接层（fc），将输出从 1000 类（ImageNet）改为 2 类（猫 / 狗）。

| 参数 | 值 |
|------|-----|
| 模型名称 | ResNet18 |
| 输入尺寸 | 224 × 224 × 3 |
| 输出类别 | 2（猫、狗） |
| 是否预训练 | 否（从头训练） |
| 参数量 | 约 11M |

`net.py` 中也提供了 **ResNet34** 的实现，如需使用，只需将 `train.py` 和 `detect.py` 中的 `model_name` 改为 `'resnet34'` 即可。

## 项目结构

```
catdog/
├── net.py                         ← 模型定义（ResNet18 / ResNet34）
├── train.py                       ← 训练脚本（从零开始训练）
├── detect.py                      ← 检测脚本（图形界面选图、预测）
├── data_set/
│   └── cats and dogs/
│       ├── train/                 ← 训练集
│       ├── val/                   ← 验证集
│       └── test/                  ← 测试集
│                     
└── weight.pth                     ← 训练生成的最佳模型权重
```

## 使用方法

### 训练模型

```bash
python train.py
```

训练参数：batch_size=64，学习率 0.001，AdamW 优化器，早停 patience=5，最大 100 轮。训练完成后自动保存最佳模型到 `weight.pth`。

### 检测单张图片

```bash
python detect.py
```

运行后会弹出文件选择对话框，选择一张图片即可显示"猫"或"狗"的预测结果与置信度。

## 训练要点

- **数据增强**：随机裁剪、水平翻转、旋转、颜色抖动（仅训练集）
- **混合精度**：GPU 上自动启用 AMP（`torch.amp`），加速训练
- **学习率调度**：验证准确率不提升时自动将学习率减半（ReduceLROnPlateau）
- **梯度裁剪**：max_norm=1.0，防止梯度爆炸
- **早停机制**：连续 5 轮验证准确率不提升则提前结束训练
- **Windows 兼容**：自动检测平台，Windows 下使用主进程加载数据
