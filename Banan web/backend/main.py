from fastapi import FastAPI, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware
import tensorflow as tf
import numpy as np
from PIL import Image
import io

app = FastAPI()

# Enable CORS so frontend can call backend
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# Load your model (.h5)
model = tf.keras.models.load_model("banana_ripeness_cnn_fast.h5")

classes = ["overripe", "ripe", "rotten", "unripe"]

def preprocess(img):
    img = img.resize((128, 128))
    img = np.array(img) / 255.0
    img = np.expand_dims(img, axis=0)
    return img

@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    img_bytes = await file.read()
    img = Image.open(io.BytesIO(img_bytes)).convert("RGB")

    processed = preprocess(img)
    prediction = model.predict(processed)[0]
    class_id = np.argmax(prediction)
    confidence = float(np.max(prediction))

    return {
        "class": classes[class_id],
        "confidence": confidence
    }
