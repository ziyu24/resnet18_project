"""
Created on 11 18, 2024
@author: <Cui>
@bref: 创建 tensorrt 来使用 onnx，该文件在 orin 下面可以正常正确运行
"""

import tensorrt as trt

import cv2

import pycuda.driver as cuda
import pycuda.autoinit
import numpy as np
import pycuda.driver as cuda
import pycuda.autoinit
import numpy as np
import torch
from torchvision.transforms.functional import to_pil_image
import matplotlib.pyplot as plt


def onnx_to_tensorrt(onnx_path, trt_path, fp16=False, max_workspace_size=1 << 30):
    """
    将 ONNX 模型转换为 TensorRT 引擎。

    参数:
    - onnx_path: ONNX 文件路径 (str)
    - trt_path: 转换后的 TensorRT 引擎文件路径 (str)
    - fp16: 是否启用 FP16 模式 (bool), 默认 False
    - max_workspace_size: 最大工作区大小 (int), 默认 1GB
    """
    # 创建 TensorRT Logger
    logger = trt.Logger(trt.Logger.WARNING)
    builder = trt.Builder(logger)
    network = builder.create_network(1 << int(trt.NetworkDefinitionCreationFlag.EXPLICIT_BATCH))
    parser = trt.OnnxParser(network, logger)

    # 读取 ONNX 模型
    with open(onnx_path, "rb") as f:
        if not parser.parse(f.read()):
            for error in range(parser.num_errors):
                print(parser.get_error(error))
            raise RuntimeError("ONNX 模型解析失败。")

    # 创建 TensorRT 引擎
    config = builder.create_builder_config()
    config.max_workspace_size = max_workspace_size
    if fp16:
        config.set_flag(trt.BuilderFlag.FP16)

    engine = builder.build_engine(network, config)

    # 保存引擎
    with open(trt_path, "wb") as f:
        f.write(engine.serialize())
    print(f"TensorRT 引擎已保存到 {trt_path}")


def softmax(x):
    exp_x = np.exp(x - np.max(x))  # 防止溢出
    return exp_x / np.sum(exp_x)


def preprocess_image(image_path, input_shape):
    """
    读取并预处理图片，使其符合模型输入的要求。

    参数:
    - image_path: 图片文件路径
    - input_shape: 模型的输入形状，例如 (1, 3, 224, 224)

    返回:
    - 预处理后的图像数据 (numpy.ndarray)，形状为 (1, 3, H, W)
    """
    # 使用 OpenCV 读取图片
    img = cv2.imread(image_path)
    if img is None:
        raise ValueError(f"Image not found at {image_path}")

    # 调整图片大小为模型要求的输入尺寸
    img = cv2.resize(img, (input_shape[3], input_shape[2]))
    # 转换为 RGB 格式
    img = img[:, :, ::-1]  # BGR to RGB
    # 归一化到 [0, 1]
    img = img.astype(np.float32) / 255.0

    # 定义标准化参数（均值和标准差）
    mean = np.array([0.485, 0.456, 0.406], dtype=np.float32)
    std = np.array([0.229, 0.224, 0.225], dtype=np.float32)

    # 逐通道减去均值并除以标准差
    img = (img - mean) / std

    # 转换为 NCHW 格式（通道、高度、宽度）
    img = np.transpose(img, (2, 0, 1))
    # 增加 batch 维度
    img = np.expand_dims(img, axis=0)

    return np.ascontiguousarray(img, dtype=np.float32)


def infer_with_tensorrt(trt_path, image_path):
    """
    使用 TensorRT 引擎进行推理，输入为图片路径。
    """
    # 加载 TensorRT 引擎
    with open(trt_path, "rb") as f:
        runtime = trt.Runtime(trt.Logger(trt.Logger.WARNING))
        engine = runtime.deserialize_cuda_engine(f.read())

    # 创建上下文
    context = engine.create_execution_context()
    bindings = []
    inputs, outputs = [], []

    # 配置缓冲区
    for i in range(engine.num_bindings):
        tensor_name = engine.get_tensor_name(i)
        tensor_shape = engine.get_tensor_shape(tensor_name)
        dtype = trt.nptype(engine.get_tensor_dtype(tensor_name))
        buffer = cuda.mem_alloc(trt.volume(tensor_shape) * np.dtype(dtype).itemsize)
        bindings.append(int(buffer))

        if engine.get_tensor_mode(tensor_name) == trt.TensorIOMode.INPUT:
            inputs.append(buffer)
        else:
            outputs.append(buffer)

    # 获取输入形状
    input_shape = engine.get_tensor_shape(engine.get_tensor_name(0))  # 假设第 0 个绑定为输入
    input_data = preprocess_image(image_path, input_shape)

    # 将输入数据复制到 GPU
    cuda.memcpy_htod(inputs[0], input_data)

    # 执行推理
    context.execute_v2(bindings)

    # 获取输出
    output_shape = engine.get_tensor_shape(engine.get_tensor_name(1))  # 假设第 1 个绑定为输出
    output_data = np.empty(output_shape, dtype=np.float32)
    cuda.memcpy_dtoh(output_data, outputs[0])

    probabilities = softmax(output_data[0])
    predicted_class = np.argmax(probabilities)

    return predicted_class


def display_inference_results_tensorrt(model, test_data, trt_path, show_number=5, device=None):
    """
    使用 TensorRT 推理并动态显示结果。

    参数:
    - model: 原始 PyTorch 模型，用于数据预处理 (torch.nn.Module)
    - test_data: 测试数据集 (torch.utils.data.DataLoader)
    - trt_path: TensorRT 引擎文件路径 (str)
    - show_number: 显示结果的数量 (int), 默认 5
    - device: PyTorch 运行设备, 默认自动选择 (torch.device)
    """
    # 选择设备
    if device is None:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # 计数器
    displayed_count = 0

    for images, labels in test_data:
        if displayed_count >= show_number:
            break

        # 预处理输入
        images = images.to(device).numpy()

        # 使用 TensorRT 进行推理
        predictions = infer_with_tensorrt(trt_path, images)

        # 显示每张图片及其预测结果
        for i in range(images.shape[0]):
            if displayed_count >= show_number:
                return

            image = to_pil_image(images[i])
            predicted_label = np.argmax(predictions[i])
            true_label = labels[i].item()

            # 显示结果
            plt.figure(figsize=(4, 4))
            plt.imshow(image)
            plt.title(f"Prediction: {predicted_label}, True: {true_label}")
            plt.axis('off')
            plt.show()

            displayed_count += 1


if __name__ == "__main__":
    # 替换为实际的 TensorRT 引擎文件路径和图片路径
    trt_engine_path = "./resnet18_self.trt"
    image_path_1 = "8_5_145_10487.jpg"
    image_path_2 = "16_3_42_12274.jpg"
    image_path_3 = "20_6_38_11264.jpg"

    try:
        output = infer_with_tensorrt(trt_engine_path, image_path_1)
        print("Inference output 8:", output)
        output = infer_with_tensorrt(trt_engine_path, image_path_2)
        print("Inference output 16:", output)
        output = infer_with_tensorrt(trt_engine_path, image_path_3)
        print("Inference output 20:", output)
    except Exception as e:
        print("Error during inference:", e)