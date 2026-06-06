"""
检测脚本
-------
加载训练好的模型权重，通过图形界面选择一张图片，
然后对图片进行猫/狗分类预测，并显示预测结果和置信度。
"""

import os
import torch
import torchvision.transforms as transforms
from PIL import Image
import tkinter as tk
from tkinter import filedialog

import net


def load_model(model_path, model_name='resnet18'):
    """
    从本地文件加载训练好的模型权重。

    参数:
        model_path (str): 模型权重文件（.pth）的路径。
        model_name (str): 模型名称，需与训练时保持一致。

    返回:
        model (nn.Module): 加载了权重并设为评估模式的模型。
        None: 当权重文件不存在时返回 None。
    """
    if os.path.exists(model_path):
        # 先创建与训练时结构一致的模型，再加载权重
        model = net.create_model(model_name, num_classes=2, pretrained=False)
        model.load_state_dict(torch.load(model_path, map_location='cpu'))
        model.eval()                # 设为评估模式
        return model
    return None


def predict_image(model, image_path):
    """
    对单张图片进行猫/狗分类预测。

    参数:
        model (nn.Module): 已加载权重的模型。
        image_path (str): 待预测图片的路径。

    返回:
        tuple: (类别名称, 置信度百分比)
            - 类别名称: "猫" 或 "狗"
            - 置信度: 0~100 的浮点数
    """
    # 与验证集相同的预处理流程
    transform = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(
            (0.485, 0.456, 0.406), (0.229, 0.224, 0.225)
        ),
    ])

    # 打开图片，统一转为 RGB 三通道
    image = Image.open(image_path).convert('RGB')
    # 增加 batch 维度（模型期望输入形状为 [batch_size, C, H, W]）
    image_tensor = transform(image).unsqueeze(dim=0)

    with torch.no_grad():           # 预测时无需计算梯度
        # softmax 将模型输出转为概率分布
        probabilities = torch.nn.functional.softmax(model(image_tensor), dim=1)
        # 取概率最大的类别索引
        class_id = torch.argmax(probabilities)

    class_names = ["猫", "狗"]
    # 取出对应类别的置信度并转为百分比
    confidence = probabilities[0][class_id].item() * 100
    return class_names[class_id], confidence


def main():
    print("猫狗分类检测器")
    print("=" * 30)

    model_name = 'resnet18'
    model_path = os.path.join(os.getcwd(), "weight.pth")
    model = load_model(model_path, model_name)

    # 模型文件不存在时，提示用户先训练
    if not model:
        print(f"模型文件不存在: {model_path}")
        print("请先运行 train.py 训练模型")
        input("按任意键退出...")
        return

    print(f"使用模型: {model_name}（本地权重）")
    print("模型加载成功")

    # ----- 文件选择对话框 -----
    # 创建一个隐藏的 tkinter 根窗口（只用于弹出文件选择框）
    root = tk.Tk()
    root.withdraw()

    # 默认打开测试集目录（如果存在）
    initial_dir = os.path.join(os.getcwd(), "data_set", "cats and dogs", "test")
    if not os.path.exists(initial_dir):
        initial_dir = os.getcwd()

    print("\n请选择测试图片...")
    file_path = filedialog.askopenfilename(
        initialdir=initial_dir,
        title="选择测试图片",
        filetypes=[
            ("图片文件", "*.jpg;*.jpeg;*.png"),
            ("所有文件", "*.*"),
        ],
    )

    root.destroy()                  # 关闭 tkinter 窗口

    # 用户取消了文件选择
    if not file_path:
        print("\n未选择任何文件")
        input("按任意键退出...")
        return

    # ----- 执行预测 -----
    print(f"\n正在检测: {os.path.basename(file_path)}")
    result, confidence = predict_image(model, file_path)

    # ----- 显示结果 -----
    print("\n" + "=" * 30)
    print(f"预测结果: {result}")
    print(f"置信度: {confidence:.2f}%")
    print("=" * 30)


if __name__ == "__main__":
    main()
