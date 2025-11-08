import torch
import torch.nn as nn
from torchvision import models, transforms
from fastapi import FastAPI, File, UploadFile
from fastapi.responses import JSONResponse
from PIL import Image
import io
import cv2
import numpy as np

app = FastAPI()

class BreastCancerCNN(nn.Module):
    def __init__(self, num_classes=3):  # update to match config.NUM_CLASSES if different
        super(BreastCancerCNN, self).__init__()

        # Load pretrained ResNet50
        self.backbone = models.resnet50(weights="IMAGENET1K_V1")  # replaces pretrained=True

        # Freeze earlier layers (transfer learning)
        for param in list(self.backbone.parameters())[:-20]:
            param.requires_grad = False

        # Replace the FC head
        in_features = self.backbone.fc.in_features
        self.backbone.fc = nn.Sequential(
            nn.Dropout(0.5),
            nn.Linear(in_features, 512),
            nn.ReLU(inplace=True),
            nn.BatchNorm1d(512),
            nn.Dropout(0.3),
            nn.Linear(512, 256),
            nn.ReLU(inplace=True),
            nn.BatchNorm1d(256),
            nn.Dropout(0.2),
            nn.Linear(256, num_classes)
        )

    def forward(self, x):
        return self.backbone(x)


# ✅ Instantiate model
model = BreastCancerCNN(num_classes=3)
num_classes = 3 # 'benigno', 'normal', 'maligno'

# load pre-trained model
MODEL_PATH = "models/cnn.pth"
model.load_state_dict(torch.load(MODEL_PATH, map_location="cpu")) # device info gpu -> cpu
model.eval()

#transform to expected format
transform = transforms.Compose([
    transforms.Resize((224,224)),

    transforms.ToTensor(),

    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    )
])

# separate the breast tissue from the background to improve classification accuracy
def segment_breast_region(image_rgb: np.ndarray) -> np.ndarray:
    try:
        gray = cv2.cvtColor(image_rgb, cv2.COLOR_RGB2GRAY)
        blurred = cv2.GaussianBlur(gray, (5, 5), 0)
        thresh = cv2.adaptiveThreshold(
            blurred,
            255,
            cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
            cv2.THRESH_BINARY,
            11,
            2,
        )
        kernel = np.ones((5, 5), np.uint8)
        cleaned = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel)
        cleaned = cv2.morphologyEx(cleaned, cv2.MORPH_OPEN, kernel)
        contours, _ = cv2.findContours(cleaned, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if contours:
            largest_contour = max(contours, key=cv2.contourArea)
            mask = np.zeros_like(gray)
            cv2.fillPoly(mask, [largest_contour], 255)
            segmented = cv2.bitwise_and(image_rgb, image_rgb, mask=mask)
            return segmented
        return image_rgb
    except Exception:
        return image_rgb

# Preprocess ToPILImage -> Resize -> ToTensor -> Normalize
preprocess = transforms.Compose([
    transforms.ToPILImage(),
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

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
    
    classes = ['Benigno', 'Maligno', 'Normal']
    label = classes[pred_class] if 0 <= pred_class < len(classes) else str(pred_class)

    return JSONResponse({
        'result': int(pred_class),
        'label': label,
        'confidence': float(confidence)
    })