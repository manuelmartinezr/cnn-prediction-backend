import torch
import torch.nn as nn
from torchvision import models, transforms
from fastapi import FastAPI, File, UploadFile
from fastapi.responses import JSONResponse
from PIL import Image
import io

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

@app.post('/predict')
async def predict(file: UploadFile = File(...)):
    img_bytes = await file.read()
    img = Image.open(io.BytesIO(img_bytes)).convert('RGB')

    input = transform(img).unsqueeze(0) # unsqueeze changes img shape 
                                        # [3, 224, 224] → [1, 3, 224, 224]
    
    with torch.no_grad(): #resource management
        outputs = model(input)
        probs = torch.nn.functional.softmax(outputs, dim=1)
        pred_class = torch.argmax(probs, dim=1).item()
        confidence = probs[0][pred_class].item()
    
    return JSONResponse({
        'result': int(pred_class),
        'confidence': float(confidence)
    })

