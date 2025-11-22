import torch
import torch.nn as nn
from app import model

dummy = torch.randn(1, 3, 224, 224)
out = model(dummy)
print(out.shape)
