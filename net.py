"""
模型定义模块
-----------
使用 torchvision 预定义的 ResNet 架构，修改最后的全连接层，
适配猫狗二分类任务（2 个类别）。
"""

import torch.nn as nn
from torchvision import models


def create_model(model_name='resnet18', num_classes=2, pretrained=False):
    """
    创建一个用于猫狗分类的 ResNet 模型。

    参数:
        model_name (str): 模型名称，可选 'resnet18' 或 'resnet34'。
        num_classes (int): 分类的类别数量，默认 2（猫和狗）。
        pretrained (bool): 是否使用在 ImageNet 上预训练的权重。

    返回:
        model (nn.Module): 构造好的 PyTorch 模型。

    异常:
        ValueError: 当 model_name 不是 'resnet18' 或 'resnet34' 时抛出。
    """

    # ----- 选择 resnet18 -----
    if model_name == 'resnet18':
        # 新版 torchvision 使用 weights 参数，旧版使用 pretrained 参数
        # 这里用 try/except 兼容两种写法
        try:
            model = models.resnet18(
                weights=models.ResNet18_Weights.DEFAULT if pretrained else None
            )
        except (TypeError, AttributeError):
            model = models.resnet18(pretrained=pretrained)

        # 替换最后的全连接层，输出类别数改为 num_classes
        model.fc = nn.Linear(model.fc.in_features, num_classes)
        return model

    # ----- 选择 resnet34 -----
    elif model_name == 'resnet34':
        try:
            model = models.resnet34(
                weights=models.ResNet34_Weights.DEFAULT if pretrained else None
            )
        except (TypeError, AttributeError):
            model = models.resnet34(pretrained=pretrained)

        # 替换最后的全连接层，输出类别数改为 num_classes
        model.fc = nn.Linear(model.fc.in_features, num_classes)
        return model

    # ----- 不支持的模型 -----
    raise ValueError(f"不支持的模型: {model_name}")
