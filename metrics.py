from typing import Dict, Optional

import numpy as np
import torch
from sklearn.metrics import confusion_matrix


def _to_numpy_labels(values):
    if isinstance(values, torch.Tensor):
        values = values.detach().cpu()
        if values.ndim > 1:
            values = torch.argmax(values, dim=1)
        return values.numpy()
    values = np.asarray(values)
    if values.ndim > 1:
        values = np.argmax(values, axis=1)
    return values


def compute_metrics(
    outputs,
    targets,
    n_classes: Optional[int] = None,
    ignored_label: int = -1,
) -> Dict[str, object]:
    """Compute HSI classification metrics in the DCT-IML dict style."""
    prediction = _to_numpy_labels(outputs).reshape(-1)
    target = _to_numpy_labels(targets).reshape(-1)

    valid_mask = target != ignored_label
    prediction = prediction[valid_mask]
    target = target[valid_mask]

    if target.size == 0:
        raise ValueError("No valid target labels were provided for metrics.")

    if n_classes is None:
        n_classes = int(max(target.max(), prediction.max())) + 1

    cm = confusion_matrix(target, prediction, labels=list(range(n_classes)))
    total = np.sum(cm)
    correct = np.trace(cm)
    accuracy = correct / float(total) if total > 0 else 0.0

    class_acc = np.zeros(n_classes, dtype=np.float64)
    for class_idx in range(n_classes):
        class_total = np.sum(cm[class_idx, :])
        class_acc[class_idx] = cm[class_idx, class_idx] / class_total if class_total else 0.0

    pa = accuracy
    pe = (
        np.sum(np.sum(cm, axis=0) * np.sum(cm, axis=1)) / float(total * total)
        if total > 0
        else 0.0
    )
    kappa = (pa - pe) / (1.0 - pe) if not np.isclose(1.0 - pe, 0.0) else 0.0

    return {
        "Confusion matrix": cm,
        "accuracy": accuracy * 100.0,
        "Accuracy": accuracy * 100.0,
        "class acc": class_acc * 100.0,
        "AA": float(np.mean(class_acc) * 100.0),
        "Kappa": float(kappa * 100.0),
    }


def show_results(results, label_values=None, agregated=False):
    text = ""

    if agregated:
        accuracies = [r["Accuracy"] for r in results]
        aa = [r["AA"] for r in results]
        kappas = [r["Kappa"] for r in results]
        class_acc = [r["class acc"] for r in results]

        class_acc_mean = np.mean(class_acc, axis=0)
        class_acc_std = np.std(class_acc, axis=0)
        cm = np.mean([r["Confusion matrix"] for r in results], axis=0)
        text += "Agregated results :\n"
    else:
        cm = results["Confusion matrix"]
        accuracy = results["Accuracy"]
        aa = results["AA"]
        classacc = results["class acc"]
        kappa = results["Kappa"]

    text += "Confusion matrix :\n"
    text += str(cm)
    text += "---\n"

    if agregated:
        text += "Accuracy: {:.02f}+/-{:.02f}\n".format(
            np.mean(accuracies), np.std(accuracies)
        )
    else:
        text += "Accuracy : {:.02f}%\n".format(accuracy)
    text += "---\n"

    text += "class acc :\n"
    if agregated:
        for label, score, std in zip(label_values, class_acc_mean, class_acc_std):
            text += "\t{}: {:.02f}+/-{:.02f}\n".format(label, score, std)
    else:
        for label, score in zip(label_values, classacc):
            text += "\t{}: {:.02f}\n".format(label, score)
    text += "---\n"

    if agregated:
        text += "AA: {:.02f}+/-{:.02f}\n".format(np.mean(aa), np.std(aa))
        text += "Kappa: {:.02f}+/-{:.02f}\n".format(np.mean(kappas), np.std(kappas))
    else:
        text += "AA: {:.02f}%\n".format(aa)
        text += "Kappa: {:.02f}\n".format(kappa)

    print(text)
