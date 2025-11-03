import torch


def pixel_accuracy(preds, masks):
    """Acurácia pixel a pixel."""
    correct = (preds == masks).float()
    acc = correct.sum() / correct.numel()
    return acc.item()


def dice_coefficient(preds, masks, smooth=1e-5):
    """Coeficiente de Dice entre duas máscaras."""
    preds_flat = preds.view(-1)
    masks_flat = masks.view(-1)
    intersection = (preds_flat * masks_flat).sum()
    return ((2. * intersection + smooth) /
            (preds_flat.sum() + masks_flat.sum() + smooth)).item()


import evaluate as hf_evaluate
from torch import nn

def eval_mean_iou(eval_pred, num_classes=16):

    metric = hf_evaluate.load("mean_iou")
    logits, labels = eval_pred

    logits_tensor = torch.from_numpy(logits)
    logits_tensor = nn.functional.interpolate(
        logits_tensor,
        size=labels.shape[-2:],
        mode="bilinear",
        align_corners=False,
    )

    pred_labels = logits_tensor.argmax(dim=1).cpu().numpy()

    metrics = metric._compute(
        predictions=pred_labels,
        references=labels,
        num_labels=num_classes,
        ignore_index=0,
        reduce_labels=False,
    )

    per_cat_acc = metrics.pop("per_category_accuracy").tolist()
    per_cat_iou = metrics.pop("per_category_iou").tolist()
    metrics.update({f"accuracy_{i}": v for i, v in enumerate(per_cat_acc)})
    metrics.update({f"iou_{i}": v for i, v in enumerate(per_cat_iou)})

    return metrics


