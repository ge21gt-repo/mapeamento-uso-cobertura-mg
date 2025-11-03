import torch
import torch.nn as nn
from torchvision import models
from src.data.loader import Data
from .base import BaseWrapper

class DeeplabWrapper(BaseWrapper):
    def __init__(self, model_path, device, num_classes=16, num_channels=12):
        deeplab = models.segmentation.deeplabv3_resnet50(weights=None)
        deeplab.backbone.conv1 = nn.Conv2d(num_channels, 64, 7, 2, 3, bias=False)
        deeplab.classifier[4] = nn.Conv2d(256, num_classes, 1)
        if deeplab.aux_classifier:
            deeplab.aux_classifier[4] = nn.Conv2d(256, num_classes, 1)
        deeplab.load_state_dict(torch.load(model_path, map_location=device))
        deeplab.to(device).eval()
        super().__init__(deeplab, device)

    def predict(self, img_path: str):
        data = Data(img_path)
        tensor, _ = data.tensor()
        with torch.no_grad():
            output = self.model(tensor.to(self.device))['out']
        return output.argmax(dim=1).squeeze().cpu().numpy()
