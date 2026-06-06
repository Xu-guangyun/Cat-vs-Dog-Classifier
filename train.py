"""
训练脚本
-------
使用迁移学习中的 ResNet18 架构，从头训练一个猫狗二分类模型。
训练过程中会进行以下操作：
  1. 数据增强（随机裁剪、翻转、旋转、颜色抖动）
  2. 混合精度训练（在支持 CUDA 的设备上自动启用，加速训练）
  3. 每 5 轮进行一次验证，评估模型在验证集上的准确率
  4. 学习率自动衰减（当验证准确率不再提升时降低学习率）
  5. 早停机制（连续 patience 轮验证准确率不提升则自动停止）
  6. 自动保存验证准确率最高的模型权重到 weight.pth
"""

import os
import sys

import torch
from torch import nn, optim
import torchvision.transforms as transforms
from torchvision import datasets
from torch.amp import GradScaler, autocast
from tqdm import tqdm

import net


def main():
    # ============================================================
    # 1. 性能优化设置
    # ============================================================
    # 开启 cuDNN 自动优化，在输入尺寸固定时能自动寻找最快的卷积算法
    torch.backends.cudnn.benchmark = True

    # ============================================================
    # 2. 数据预处理与增强
    # ============================================================
    data_transform = {
        # 训练集：使用多种数据增强手段，提升模型的泛化能力
        "train": transforms.Compose([
            transforms.RandomResizedCrop(224, scale=(0.8, 1.0)),   # 随机裁剪并缩放至 224×224
            transforms.RandomHorizontalFlip(),                     # 随机水平翻转
            transforms.RandomRotation(15),                         # 随机旋转 ±15 度
            transforms.ColorJitter(                                # 随机调整亮度、对比度、饱和度
                brightness=0.2, contrast=0.2, saturation=0.2
            ),
            transforms.ToTensor(),                                 # 将 PIL 图像转为 Tensor
            transforms.Normalize(                                  # 用 ImageNet 的均值和标准差归一化
                (0.485, 0.456, 0.406), (0.229, 0.224, 0.225)
            ),
        ]),
        # 验证集：只做基础的缩放和中心裁剪，不做数据增强
        "val": transforms.Compose([
            transforms.Resize(256),                                # 短边缩放至 256
            transforms.CenterCrop(224),                            # 中心裁剪 224×224
            transforms.ToTensor(),
            transforms.Normalize(
                (0.485, 0.456, 0.406), (0.229, 0.224, 0.225)
            ),
        ]),
    }

    # ============================================================
    # 3. 加载数据集
    # ============================================================
    data_root = os.path.join(os.getcwd(), "data_set")
    print("数据集根目录：", data_root)

    dataset_path = os.path.join(data_root, "cats and dogs")
    print("数据集目录：", dataset_path)

    # --- 训练集 ---
    train_dataset = datasets.ImageFolder(
        root=os.path.join(dataset_path, "train"),
        transform=data_transform["train"]
    )
    train_num = len(train_dataset)
    print("训练集图片数量：", train_num)

    # --- 配置 DataLoader ---
    batch_size = 64

    # Windows 下多进程数据加载容易出问题，因此设为 0（主进程加载）
    # Linux/macOS 下使用多进程加速数据读取
    num_workers = 0 if sys.platform == "win32" else min(
        [os.cpu_count(), batch_size if batch_size > 1 else 0, 8]
    )
    if num_workers > 0:
        print(f'正在使用 {num_workers} 个子进程加载数据集')
    else:
        print('使用主进程（同步）加载数据集')

    train_loader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,               # 训练时打乱数据顺序
        num_workers=num_workers,
        pin_memory=True,            # 锁页内存，加速 GPU 数据传输
    )
    print("训练集加载完毕\n开始加载验证集")

    # --- 验证集 ---
    validate_dataset = datasets.ImageFolder(
        root=os.path.join(dataset_path, "val"),
        transform=data_transform["val"]
    )
    val_num = len(validate_dataset)
    print("验证集图片数量：", val_num)

    validate_loader = torch.utils.data.DataLoader(
        validate_dataset,
        batch_size=64,
        shuffle=False,              # 验证时不需要打乱
        num_workers=num_workers,
        pin_memory=True,
    )

    # ============================================================
    # 4. 设备配置（GPU / CPU）
    # ============================================================
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    if torch.cuda.is_available():
        device_count = torch.cuda.device_count()
        print(f"找到 {device_count} 个CUDA设备:")
        for i in range(device_count):
            print(f"  GPU {i}: {torch.cuda.get_device_name(i)}")
        print(f"当前使用: {device}")
    else:
        print("没有找到CUDA设备，使用CPU训练")

    # ============================================================
    # 5. 创建模型
    # ============================================================
    model_name = 'resnet18'
    model = net.create_model(model_name, num_classes=2, pretrained=False)
    model.to(device)                    # 将模型移动到 GPU/CPU
    print(f"使用模型: {model_name}（从头训练）")

    # ============================================================
    # 6. 损失函数、优化器、学习率调度器
    # ============================================================
    loss_function = nn.CrossEntropyLoss()   # 交叉熵损失，适用于分类任务
    optimizer = optim.AdamW(                # AdamW 优化器，带权重衰减
        model.parameters(), lr=0.001, weight_decay=1e-4
    )
    # 当验证准确率不再提升时，将学习率缩小为原来的 factor 倍
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='max', factor=0.5, patience=5
    )

    # ============================================================
    # 7. 混合精度训练（仅 CUDA 设备可用）
    # ============================================================
    # GradScaler 用于缩放梯度，防止 float16 精度下的小梯度被当作零
    scaler = GradScaler('cuda') if device.type == 'cuda' else None

    # 权重保存路径
    weight_save_path = 'weight.pth'
    print("权重文件保存路径：", os.path.join(os.getcwd(), weight_save_path))

    # ============================================================
    # 8. 训练超参数
    # ============================================================
    epochs = 100                        # 最大训练轮数
    best_accuracy = 0.0                 # 记录最佳验证准确率
    steps_per_epoch = len(train_loader) # 每轮的批次数

    # 早停机制：连续 patience 轮验证准确率没有提升则停止训练
    patience = 5
    stagnant_epochs = 0                 # 计数器（未提升的轮数）

    # ============================================================
    # 9. 训练循环
    # ============================================================
    model.train()
    for epoch in range(epochs):
        running_loss = 0.0
        # tqdm 进度条，显示当前批次的损失值
        train_bar = tqdm(train_loader, file=sys.stdout)

        for images, labels in train_bar:
            # 将数据移动到 GPU/CPU（non_blocking=True 异步传输，配合 pin_memory 提速）
            images = images.to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True)

            optimizer.zero_grad()       # 清空上一轮的梯度

            # --- 混合精度训练分支 ---
            if scaler:
                with autocast('cuda'):  # 自动将部分计算转为 float16
                    outputs = model(images)
                    loss = loss_function(outputs, labels)
                scaler.scale(loss).backward()       # 对缩放后的 loss 反向传播
                scaler.unscale_(optimizer)           # 还原梯度，以便裁剪
                torch.nn.utils.clip_grad_norm_(      # 梯度裁剪，防止梯度爆炸
                    model.parameters(), max_norm=1.0
                )
                scaler.step(optimizer)               # 更新参数
                scaler.update()                      # 更新缩放因子
            # --- 普通训练分支（CPU 或非 CUDA 设备）---
            else:
                outputs = model(images)
                loss = loss_function(outputs, labels)
                loss.backward()                      # 反向传播
                torch.nn.utils.clip_grad_norm_(      # 梯度裁剪
                    model.parameters(), max_norm=1.0
                )
                optimizer.step()                     # 更新参数

            # 累计损失值（用于后续计算平均损失）
            running_loss += loss.item()
            train_bar.desc = "train epoch[{}/{}] loss:{:.3f}".format(
                epoch + 1, epochs, loss
            )

        # ========================================================
        # 10. 验证阶段（每 5 轮 或 最后一轮）
        # ========================================================
        if (epoch % 5 == 0 and epoch > 1) or epoch == epochs - 1:
            model.eval()                # 切换到评估模式（关闭 Dropout 等）
            val_correct = 0

            with torch.no_grad():       # 验证时不需要计算梯度
                val_bar = tqdm(validate_loader, file=sys.stdout)
                for val_images, val_labels in val_bar:
                    val_images = val_images.to(device, non_blocking=True)
                    val_labels = val_labels.to(device, non_blocking=True)

                    outputs = model(val_images)
                    # 取概率最大的类别作为预测结果
                    pred_labels = torch.max(outputs, dim=1)[1]
                    # 累加预测正确的数量
                    val_correct += torch.eq(
                        pred_labels, val_labels
                    ).sum().item()

            # 计算验证准确率
            val_accuracy = val_correct / val_num
            print('[epoch %d] train_loss: %.3f  val_accuracy: %.3f' %
                  (epoch + 1, running_loss / steps_per_epoch, val_accuracy))

            # 更新学习率调度器（根据验证准确率决定是否降低学习率）
            scheduler.step(val_accuracy)

            # --- 保存最佳模型 ---
            if val_accuracy > best_accuracy:
                best_accuracy = val_accuracy
                torch.save(model.state_dict(), weight_save_path)
                print(f"保存最佳模型，准确率: {val_accuracy:.3f}")
                stagnant_epochs = 0
            else:
                stagnant_epochs += 1
                print(f"连续 {stagnant_epochs}/{patience} 轮没有提升")
                # 早停：连续多轮未提升则提前结束训练
                if stagnant_epochs >= patience:
                    print(f"\n连续{patience}轮验证准确率没有提升，提前停止训练")
                    print(f"最佳准确率: {best_accuracy:.3f}")
                    break

            model.train()               # 切回训练模式，继续下一轮训练

    print('训练完成')


if __name__ == '__main__':
    main()
