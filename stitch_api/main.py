from fastapi import FastAPI, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware
from PIL import Image
import numpy as np
import onnxruntime as ort
import io

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

session = ort.InferenceSession("best.onnx", providers=["CPUExecutionProvider"])

labels = ["长针绣", "平针绣", "乱针绣", "十字绣", "其他"]

def preprocess(image: Image.Image) -> np.ndarray:
    image = image.resize((224, 224))
    image = np.array(image).astype(np.float32) / 255.0
    if image.shape[-1] == 4:
        image = image[..., :3]
    image = np.transpose(image, (2, 0, 1))
    return image[np.newaxis, :]

@app.post("/predict_by_file")
async def predict_by_file(file: UploadFile = File(...)):
    try:
        contents = await file.read()
        image = Image.open(io.BytesIO(contents)).convert("RGB")
        input_tensor = preprocess(image)
        outputs = session.run(None, {"images": input_tensor})
        probs = outputs[0][0]
        pred_class = int(np.argmax(probs))
        label = labels[pred_class]
        return {
            "predicted_class": label,
            "confidence": float(np.max(probs))
        }
    except Exception as e:
        return {"error": f"处理图片时出错: {str(e)}"}
