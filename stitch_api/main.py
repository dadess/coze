from flask import Flask, request, jsonify
import onnxruntime as ort
import numpy as np
from PIL import Image
import cv2
import io

app = Flask(__name__)

# 加载 ONNX 模型
session = ort.InferenceSession("best.onnx")
input_name = session.get_inputs()[0].name
output_name = session.get_outputs()[0].name

# 图片预处理函数（根据 YOLOv8 设置调整）
def preprocess_image(image_bytes):
    image = Image.open(io.BytesIO(image_bytes)).convert("RGB")
    image = image.resize((640, 640))  # 根据训练时的尺寸
    img = np.array(image).astype(np.float32)
    img = img / 255.0
    img = np.transpose(img, (2, 0, 1))  # HWC -> CHW
    img = np.expand_dims(img, axis=0)  # Add batch dim
    return img

# 后处理函数（你可根据实际类别数和置信度阈值修改）
def postprocess_output(output):
    predictions = output[0]
    # 示例：只取类别编号，实际可根据类别名映射
    pred_class = int(np.argmax(predictions))
    return pred_class

@app.route('/predict', methods=['POST'])
def predict():
    if 'image' not in request.files:
        return jsonify({'error': 'No image uploaded'}), 400

    image_file = request.files['image']
    image_bytes = image_file.read()
    input_tensor = preprocess_image(image_bytes)

    output = session.run([output_name], {input_name: input_tensor})
    result = postprocess_output(output)
    
    # 映射类别编号为刺绣针法名称（你需要自己写好字典）
    class_map = {
        0: "平针",
        1: "锁针",
        2: "回针",
        3: "交叉针",
        # 添加更多针法...
    }

    return jsonify({'class': result, 'stitch': class_map.get(result, "未知")})
