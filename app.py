import torch
import torch.nn as nn
from torchvision import models, transforms
from fastapi import FastAPI, File, UploadFile
from fastapi.responses import JSONResponse
from PIL import Image
import io
import cv2
import numpy as np
from fastapi.middleware.cors import CORSMiddleware

from models.cnn_model import load_model
from utils.image_processing import segment_breast_region, preprocess

app = FastAPI()

# enable cors for local testing
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],  
)

# initialize model
model = load_model()
classes = ['Benigno', 'Maligno', 'Normal']

@app.post('/predict')
async def predict(file: UploadFile = File(...)):
    img_bytes = await file.read()
    img = Image.open(io.BytesIO(img_bytes)).convert('RGB')
    img_np = np.array(img)
    img_np = segment_breast_region(img_np)
    input = preprocess(img_np).unsqueeze(0) 
    
    with torch.no_grad(): 
        outputs = model(input)
        probs = torch.nn.functional.softmax(outputs, dim=1)
        pred_class = torch.argmax(probs, dim=1).item()
        confidence = probs[0][pred_class].item()
    
    label = classes[pred_class] if 0 <= pred_class < len(classes) else str(pred_class)

    return JSONResponse({
        'result': int(pred_class),
        'label': label,
        'confidence': float(confidence)
    })