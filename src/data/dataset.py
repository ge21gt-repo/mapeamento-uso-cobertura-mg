import os
import glob
import numpy as np
from osgeo import gdal
from torch.utils.data import Dataset
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import albumentations as A
import tensorflow as tf


class RemoteSensingDataset(Dataset):
    def __init__(self, image_paths, mask_paths, feature_extractor, num_classes, mode):
        self.image_paths = image_paths
        self.mask_paths = mask_paths
        self.feature_extractor = feature_extractor
        self.num_classes = num_classes
        self.mode = mode

    def __getitem__(self, idx):
        if self.mode == "multiband":
            return self.__multibanditem__(idx)
        elif self.mode == "rgb":
            return self.__rgbitem__(idx)
        else:
            raise ValueError(f"Modo inválido: {self.mode}")

    def __len__(self):
        return len(self.image_paths)

    def __multibanditem__(self, idx):
        # Carrega imagem (12 bandas)
        img_ds = gdal.Open(self.image_paths[idx])
        img = img_ds.ReadAsArray().astype(np.float32)
        img_ds = None

        # Seleciona bandas RGB
        if img.shape[0] >= 3:
            img = img[:3]
        else:
            raise ValueError("A imagem não possui ao menos 3 bandas RGB.")

        # Normalização por banda
        for i in range(img.shape[0]):
            band = img[i]
            min_val = band.min()
            max_val = band.max()
            if max_val - min_val > 1e-8:
                img[i] = (band - min_val) / (max_val - min_val)
            else:
                img[i] = band * 0

        img = np.transpose(img, (1, 2, 0))  # (H, W, C)

        # Carrega máscara
        mask_ds = gdal.Open(self.mask_paths[idx])
        mask = mask_ds.ReadAsArray().astype(np.int64)
        mask_ds = None

        if len(mask.shape) == 3:
            mask = mask.squeeze(0)

        # Pré-processamento
        inputs = self.feature_extractor(
            images=img,
            segmentation_maps=mask,
            return_tensors="pt",
        )
        inputs = {k: v.squeeze(0) for k, v in inputs.items()}
        return inputs

    def __rgbitem__(self, idx):
        # Carrega imagem
        img_ds = gdal.Open(self.image_paths[idx])
        img = img_ds.ReadAsArray().astype(np.float32)
        img_ds = None

        # Normalização por banda
        for i in range(img.shape[0]):
            band = img[i]
            min_val = band.min()
            max_val = band.max()
            if max_val - min_val > 1e-8:
                img[i] = (band - min_val) / (max_val - min_val)
            else:
                img[i] = band * 0  # Caso banda constante

        # Carrega máscara
        mask_ds = gdal.Open(self.mask_paths[idx])
        mask = mask_ds.ReadAsArray().astype(np.int64)
        mask_ds = None

        # Processamento da máscara:
        # - Valores fora do intervalo 0-15 são marcados como -1 (ignorar)
        new_mask = np.full_like(mask, -1)  # Inicializa tudo como -1
        valid_mask = (mask >= 0) & (mask < self.num_classes)
        new_mask[valid_mask] = mask[valid_mask]

        # Garante que máscara tem dimensões corretas
        if len(new_mask.shape) == 3:
            new_mask = new_mask.squeeze(0)

        return img, new_mask

    def plot_sample(self, idx):
        """Plota uma imagem RGB normalizada e a máscara correspondente."""
        # Carrega imagem e máscara originais
        img_ds = gdal.Open(self.image_paths[idx])
        img = img_ds.ReadAsArray().astype(np.float32)
        img_ds = None

        if img.shape[0] >= 3:
            img = img[:3]
        else:
            raise ValueError("A imagem não possui ao menos 3 bandas RGB.")

        for i in range(img.shape[0]):
            band = img[i]
            min_val = band.min()
            max_val = band.max()
            if max_val - min_val > 1e-8:
                img[i] = (band - min_val) / (max_val - min_val)
            else:
                img[i] = band * 0

        img = np.transpose(img, (1, 2, 0))  # (H, W, C)

        mask_ds = gdal.Open(self.mask_paths[idx])
        mask = mask_ds.ReadAsArray().astype(np.int64)
        mask_ds = None
        if len(mask.shape) == 3:
            mask = mask.squeeze(0)

        # Plotagem
        fig, axs = plt.subplots(1, 2, figsize=(12, 6))

        axs[0].imshow(img)
        axs[0].set_title("Imagem RGB")
        axs[0].axis("off")

        cmap = plt.get_cmap("tab20", self.num_classes)
        im = axs[1].imshow(mask, cmap=cmap, vmin=0, vmax=16)
        axs[1].set_title("Máscara de Segmentação")
        axs[1].axis("off")

        cbar = fig.colorbar(im, ax=axs, orientation='horizontal', fraction=0.05, pad=0.05)
        cbar.set_ticks(list(range(self.num_classes)))
        cbar.set_label("Classes (0 a 15)")

        plt.tight_layout()
        plt.show()



def load_pairs_torch(img_dir, mask_dir):
    print("Preparando dados...")
    image_files = sorted(glob.glob(os.path.join(img_dir, "*.tif")))
    mask_files = []

    for img_path in image_files:
        base_name = os.path.basename(img_path).replace('.tif', '')
        mask_path = os.path.join(mask_dir, f"{base_name}_mask.tif")

        if not os.path.exists(mask_path):
            # Tenta encontrar máscara com padrão alternativo
            alt_mask = glob.glob(os.path.join(mask_dir, f"{base_name}*_mask.tif"))
            if alt_mask:
                mask_path = alt_mask[0]

        if os.path.exists(mask_path):
            mask_files.append(mask_path)
        else:
            print(f"Aviso: Máscara não encontrada para {img_path}")

    # Filtra imagens sem máscaras correspondentes
    valid_pairs = [(img, mask) for img, mask in zip(image_files, mask_files) if os.path.exists(mask)]
    image_files, mask_files = zip(*valid_pairs) if valid_pairs else ([], [])

    # Divisão treino/validação
    train_img, val_img, train_mask, val_mask = train_test_split(
        list(image_files), list(mask_files), test_size=0.25, random_state=42
    )

    print(f"Total de amostras: {len(image_files)}")
    print(f"Treino: {len(train_img)} | Validação: {len(val_img)}")
    return train_img, train_mask, val_img, val_mask
