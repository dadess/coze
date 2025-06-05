from fastapi import FastAPI, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware
from PIL import Image
import numpy as np
import onnxruntime as ort
import io

app = FastAPI()

# 开放 CORS 权限（给 Coze 调用）
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # 可以指定为 Coze 的插件调用域名
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# 加载模型
session = ort.InferenceSession("best.onnx", providers=["CPUExecutionProvider"])

# 假设模型的输入为 (1, 3, 224, 224)，输出为 [1, num_classes]
def preprocess(image: Image.Image) -> np.ndarray:
    image = image.resize((224, 224))
    image = np.array(image).astype(np.float32) / 255.0
    if image.shape[-1] == 4:
        image = image[..., :3]  # 去掉 alpha 通道
    image = np.transpose(image, (2, 0, 1))  # HWC → CHW
    return image[np.newaxis, :]

# 类别名（可根据你的训练数据集设置）
labels = ["长针绣", "平针绣", "乱针绣", "十字绣", "其他"]

@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    image_bytes = await file.read()
    image = Image.open(io.BytesIO(image_bytes)).convert("RGB")
    input_tensor = preprocess(image)

    outputs = session.run(None, {"images": input_tensor})
    probs = outputs[0][0]
    pred_class = int(np.argmax(probs))
    label = labels[pred_class]

    return {
        "predicted_class": label,
        "confidence": float(np.max(probs))
    }
