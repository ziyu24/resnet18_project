"""
Created on 11 16, 2024
@author: <Cui>
@bref: 构建通用的导出 onnx 的方法，以及 onnx 推理图片的方法
"""

import torch
import onnxruntime
import numpy as np
from PIL import Image
from torchvision import transforms


def export_pth_to_onnx_common(model, pth_path, onnx_path, input_size=(1, 3, 224, 224)):
    """
    从 .pth 模型文件导出为 ONNX 格式。

    参数:
        model: PyTorch 模型结构 (需自行定义)
        pth_path: str, .pth 文件路径
        onnx_path: str, 导出的 ONNX 模型保存路径
        num_classes: int, 分类数
        input_size: tuple, 模拟的输入大小 (默认是单张 224x224 RGB 图片)

    返回:
        None
    """
    # 加载模型权重
    model.load_state_dict(torch.load(pth_path))
    model.eval()  # 设置为推理模式

    # 模拟输入张量
    dummy_input = torch.randn(*input_size)

    # 导出 ONNX 模型
    torch.onnx.export(
        model,
        dummy_input,
        onnx_path,
        input_names=["input"],
        output_names=["output"],
        dynamic_axes={"input": {0: "batch_size"}, "output": {0: "batch_size"}},
        opset_version=11  # ONNX opset version
    )
    print(f"ONNX 模型已导出到: {onnx_path}")


def infer_with_onnx_common(onnx_path, image_path):
    """
    使用 ONNXRuntime 对单张图片进行推理。

    参数:
        onnx_path: str, ONNX 模型文件路径
        image_path: str, 图片路径

    返回:
        predicted_class: int, 预测类别
    """
    # 加载 ONNX 模型
    onnx_session = onnxruntime.InferenceSession(onnx_path)

    # 获取输入输出的名称
    input_name = onnx_session.get_inputs()[0].name
    output_name = onnx_session.get_outputs()[0].name

    # 预处理图片
    def preprocess_image(image_path):
        image = Image.open(image_path).convert("RGB")
        preprocess = transforms.Compose([
            transforms.Resize((224, 224)),  # 调整图片大小
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])
        return preprocess(image).unsqueeze(0).numpy()  # 增加 batch 维度

    input_data = preprocess_image(image_path)

    # 推理
    outputs = onnx_session.run([output_name], {input_name: input_data})

    # 解析结果
    predictions = np.array(outputs).squeeze()  # 去掉 batch 维度
    predicted_class = np.argmax(predictions)
    print(f"Predicted class: {predicted_class}")
    return predicted_class

