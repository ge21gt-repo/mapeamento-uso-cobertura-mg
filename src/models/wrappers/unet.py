from src.data.loader import Data
from .base import BaseWrapper
import torch
import segmentation_models_pytorch as smp

class UnetWrapper(BaseWrapper):
    def __init__(self, model_path, device, num_classes=16, num_channels=12, encoder_name="mit_b0"):
        # Cria o modelo com o encoder especificado
        model = smp.Unet(
            encoder_name=encoder_name,      
            encoder_weights=None,
            in_channels=num_channels,
            classes=num_classes
        )

        # Carrega pesos treinados
        state_dict = torch.load(model_path, map_location=device)
        model.load_state_dict(state_dict)

        # Move o modelo para o dispositivo e modo de avaliação
        model = model.to(device).eval()

        super().__init__(model, device)

    def predict(self, img_path: str):
        data = Data(img_path)
        tensor, _ = data.tensor()  # assume que retorna (tensor, metadata)

        with torch.no_grad():
            output = self.model(tensor.to(self.device))  # sem ['out']
            pred = output.argmax(dim=1).squeeze().cpu().numpy()

        return pred
