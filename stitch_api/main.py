from fastapi import FastAPI, UploadFile, File, Query
from fastapi.middleware.cors import CORSMiddleware
from PIL import Image
import numpy as np
import onnxruntime as ort
import io
import requests

app = FastAPI()

# 开放跨域权限（允许 Coze 访问）
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # 或指定为 Coze 的域名
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# 加载 ONNX 模型
session = ort.InferenceSession("best.onnx", providers=["CPUExecutionProvider"])

# 假设模型输入为 (1, 3, 224, 224)
def preprocess(image: Image.Image) -> np.ndarray:
    image = image.resize((224, 224))
    image = np.array(image).astype(np.float32) / 255.0
    if image.shape[-1] == 4:
        image = image[..., :3]  # 去掉 alpha 通道
    image = np.transpose(image, (2, 0, 1))  # HWC → CHW
    return image[np.newaxis, :]

# 模型类别标签（根据你训练时的分类顺序）
labels = ["长针绣", "平针绣", "乱针绣", "十字绣", "其他"]

# ✅ 方法一：POST 上传图片
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
        "刺绣类别": label,
        "置信度": round(float(np.max(probs)), 4)
    }

# ✅ 方法二：GET 请求图片链接
@app.get("/predict_by_url")
def predict_by_url(img_url: str = Query(..., description="刺绣图片的URL")):
    try:
        response = requests.get(img_url)
        image = Image.open(io.BytesIO(response.content)).convert("RGB")
    except Exception as e:
        return {"error": f"无法处理图片: {str(e)}"}

    input_tensor = preprocess(image)
    outputs = session.run(None, {"images": input_tensor})
    probs = outputs[0][0]
    pred_class = int(np.argmax(probs))
    label = labels[pred_class]

    return {
        "刺绣类别": label,
        "置信度": round(float(np.max(probs)), 4)
    }
