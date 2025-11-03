import os, glob, torch,  numpy as np
from typing import Dict, Any
from src.data.loader import Data

from utils.metrics import (
    pixel_accuracy,
    dice_coefficient,
    eval_mean_iou,
)
from src.models import SegmentationModels


def to_long_tensor(x: np.ndarray) -> torch.Tensor:
    return torch.from_numpy(x.astype(np.int64))


def binarize(arr: np.ndarray, cls: int) -> torch.Tensor:
    return torch.from_numpy((arr == cls).astype(np.float32))


def one_hot(labels: np.ndarray, num_classes: int) -> np.ndarray:
    if labels.ndim == 2:
        labels = labels[None]
    n, h, w = labels.shape
    oh = np.zeros((n, num_classes, h, w), dtype=np.float32)
    for c in range(num_classes):
        oh[:, c] = (labels == c).astype(np.float32)
    return oh


def eval(
    model_type: str,
    images_dir: str = "dataset/Augmented/Images",
    masks_dir: str = "dataset/Augmented/Masks",
    num_classes: int = 16,
) -> Dict[str, Any]:
    model = SegmentationModels()
    image_paths = sorted(glob.glob(os.path.join(images_dir, "*.tif")))
    if not image_paths:
        raise RuntimeError(f"Nenhuma imagem .tif encontrada em {images_dir}")

    preds, gts = [], []
    pixel_accs, dice_macros = [], []

    # -------- Loop principal --------
    for img_path in image_paths[:100]:
        name = os.path.basename(img_path).replace(".tif", "")
        mask_path = os.path.join(masks_dir, f"{name}_mask.tif")
        if not os.path.exists(mask_path):
            print(f"[aviso] Máscara ausente para {name}, pulando.")
            continue

        # Predição por modelo
        if model_type == "deeplab":
            pred = model.deeplab.predict(img_path)
        elif model_type == "segformer":
            pred = model.segformer.predict(img_path)
        elif model_type == "unet":
            pred = model.unet.predict(img_path)
        else:
            raise ValueError("model_type deve ser 'deeplab', 'segformer' ou 'unet'.")

        pred = np.asarray(pred, dtype=np.int64)
        gt = Data(mask_path)._read()

        # Métricas universais
        pixel_accs.append(pixel_accuracy(to_long_tensor(pred), to_long_tensor(gt)))

        dices = [
            dice_coefficient(binarize(pred, c), binarize(gt, c))
            for c in range(num_classes)
        ]
        dice_macros.append(np.mean(dices))

        preds.append(pred)
        gts.append(gt)

    if not preds:
        raise RuntimeError("Nenhuma predição válida.")

    # -------- Métricas específicas --------
    mean_iou = None

    if model_type == "segformer":
        logits_oh = one_hot(np.stack(preds), num_classes)
        mean_iou = eval_mean_iou((logits_oh, np.stack(gts)), num_classes)

    if model_type == "unet":
        logits_oh = one_hot(np.stack(preds), num_classes)
        mean_iou = eval_mean_iou((logits_oh, np.stack(gts)), num_classes)

    if model_type == "deeplab":
        logits_oh = one_hot(np.stack(preds), num_classes)
        mean_iou = eval_mean_iou((logits_oh, np.stack(gts)), num_classes)


    # -------- Consolidação --------
    results = {
        "dataset_size": len(preds),
        "pixel_acc_mean": float(np.mean(pixel_accs)),
        "dice_macro_mean": float(np.mean(dice_macros)),
        "mean_iou": mean_iou,
    }

    print(f"\n=== Avaliação {model_type.upper()} ===")
    print(f"Imagens avaliadas: {results['dataset_size']}")
    print(f"Pixel Acc: {results['pixel_acc_mean']:.4f}")
    print(f"Dice Macro: {results['dice_macro_mean']:.4f}")
    print(f"Mean IOU: {results['mean_iou']['mean_iou']:.4f}")
    

    return results

if __name__ == "__main__":
    eval("segformer", num_classes=16)
