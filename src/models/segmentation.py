import warnings
warnings.filterwarnings("ignore")

import torch
from .wrappers import SegformerWrapper, DeeplabWrapper, UnetWrapper

class SegmentationModels:
    """
    Interface unificada para diferentes modelos de segmentação.
    Uso:
        model = SegmentationModels()
        mask = model.deeplab.predict(img_path)
    """
    def __init__(self,
                 segformer_path="./output/segformer_latest/final_model",
                 deeplab_path="./output/deeplab_latest/best_model.pth",
                 unet_path="./output/unet_latest/best_model.pth",
                 num_classes=16, num_channels=12):

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # Instâncias dos modelos (carregados apenas uma vez)
        self.segformer = SegformerWrapper(segformer_path, self.device)
        self.deeplab = DeeplabWrapper(deeplab_path, self.device,
                                      num_classes=num_classes, num_channels=num_channels)
        self.unet = UnetWrapper(unet_path, self.device,
                                num_classes=num_classes, num_channels=num_channels)
