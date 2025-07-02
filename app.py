import cv2
import torch
import numpy as np
import joblib
import uvicorn
from fastapi import FastAPI, File, UploadFile
from fastapi.responses import JSONResponse
from pathlib import Path
from torchvision import transforms
import os

# ✅ Load Pretrained HRNet Model (for facial landmark detection)
hrnet_model_path = "hrnetv2_w18.pth"  # Path to HRNet model weights
device = "cpu"

# Load HRNet Model
hrnet_model = torch.hub.load("HRNet/HRNet", "hrnet_w18")
hrnet_model.load_state_dict(torch.load(hrnet_model_path, map_location=device))
hrnet_model.eval().to(device)

# ✅ Load Trained Stroke Prediction Model
stroke_model = joblib.load("mlp_stroke.pkl")  # Change to "random_forest_stroke.pkl" if using RF
scaler = joblib.load("scaler.pkl")  # Load the scaler

# ✅ FastAPI Initialization
app = FastAPI()

# ✅ Preprocessing Function
def preprocess_image(image_path):
    image = cv2.imread(image_path)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    image = cv2.resize(image, (256, 256))  # Resize for HRNet
    image_tensor = transforms.ToTensor()(image).unsqueeze(0).to(device)
    return image, image_tensor

# ✅ Predict Landmarks
def predict_landmarks(image_tensor):
    with torch.no_grad():
        output = hrnet_model(image_tensor).cpu().numpy()
    landmarks = output.reshape(-1, 2) * 256  # Rescale to image size
    return landmarks.tolist()

# ✅ Predict Stroke Probability
def predict_stroke(landmarks):
    landmarks = np.array(landmarks).flatten().reshape(1, -1)  # Flatten landmarks
    landmarks_scaled = scaler.transform(landmarks)  # Normalize
    stroke_prob = stroke_model.predict_proba(landmarks_scaled)[0][1] * 100  # Get stroke probability
    return stroke_prob

# ✅ API Endpoint for Image Upload & Prediction
@app.post("/predict/")
async def predict_stroke_risk(file: UploadFile = File(...)):
    # Save uploaded file
    image_path = Path(f"uploads/{file.filename}")
    image_path.parent.mkdir(parents=True, exist_ok=True)
    with open(image_path, "wb") as buffer:
        buffer.write(await file.read())

    # Preprocess Image
    image, image_tensor = preprocess_image(str(image_path))

    # Predict Landmarks
    landmarks = predict_landmarks(image_tensor)

    # Predict Stroke Probability
    stroke_prob = predict_stroke(landmarks)

    # ✅ Send Response to Frontend
    return JSONResponse(content={"landmarks": landmarks, "stroke_probability": stroke_prob})

PORT = int(os.getenv("PORT", 8080))

# ✅ Run Server
if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=PORT)
