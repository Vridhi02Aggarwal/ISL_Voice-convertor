
from fastapi import FastAPI, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware
import cv2
import numpy as np
import json
from pathlib import Path
from app.models.opencv_model import process_frame
from tensorflow.keras.models import load_model
from tensorflow.keras.applications.efficientnet import preprocess_input

# -----------------------------
# Create FastAPI app
# -----------------------------
app = FastAPI()

# -----------------------------
# Enable CORS (IMPORTANT)
# -----------------------------
app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://localhost:8080",
        "http://127.0.0.1:8080",
        "http://localhost:3000",
        "http://127.0.0.1:3000"
    ],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# -----------------------------
# Load EfficientNet model
# -----------------------------
model_path = "app/models/isl_efficientnet_best_4.keras"
# model_path = "app/models/isl_cnn_best.h5"
cnn_model = load_model(model_path)

# -----------------------------
# Load label mapping
# -----------------------------
mapping_path = Path(__file__).parent / "data" / "label_map_reversed.json"
gesture_mapping = json.load(open(mapping_path, "r"))

gesture_index_to_label = {int(k): v for k, v in gesture_mapping.items()}

# -----------------------------
# Routes
# -----------------------------
@app.get("/")
def home():
    return "ISL Hand Gesture Recognition API"

@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    try:
        contents = await file.read()
        np_arr = np.frombuffer(contents, np.uint8)
        img = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)
        if img is None:
            return {"error": "Invalid image"}

        


        # ---- Process frame using your tracking pipeline ----
        processed, hand_img = process_frame(img)

        # If no hand found
        if hand_img is None:
            return {"error": "Hand not detected"}

        # Convert BGR → RGB
        hand_img = cv2.cvtColor(hand_img, cv2.COLOR_BGR2RGB)

        # Ensure correct shape (square crop)
        h, w, _ = hand_img.shape
        size = min(h, w)
        center_x, center_y = w // 2, h // 2

        # Square crop around the center
        x1 = center_x - size // 2
        y1 = center_y - size // 2
        x2 = x1 + size
        y2 = y1 + size
        hand_img = hand_img[y1:y2, x1:x2]

        # Resize to EfficientNet size
        hand_img = cv2.resize(hand_img, (224, 224))

        # Preprocess for EfficientNet
        hand_img = preprocess_input(hand_img.astype("float32"))
        hand_img = np.expand_dims(hand_img, axis=0)


        pred = cnn_model.predict(hand_img)
        gesture_id = int(np.argmax(pred, axis=1)[0])
        gesture_text = gesture_index_to_label.get(gesture_id, "Unknown")

        return {"gesture_id": gesture_id, "gesture_text": gesture_text}

    except Exception as e:
        return {"error": str(e)}

