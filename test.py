import torch
from datasets import load_dataset
from sklearn.metrics import (
    roc_auc_score,
    average_precision_score,
    precision_score,
    recall_score,
    f1_score
)


def compute_metrics(probs, preds, labels):
    labels_np = labels.numpy()
    preds_np = preds.numpy()
    probs_np = probs.numpy()

    auroc = roc_auc_score(labels_np, probs_np)
    aupr  = average_precision_score(labels_np, probs_np)
    precision = precision_score(labels_np, preds_np)
    recall    = recall_score(labels_np, preds_np)
    f1        = f1_score(labels_np, preds_np)

    return {
        "AUROC": auroc,
        "AUPR": aupr,
        "Precision": precision,
        "Recall": recall,
        "F1": f1
    }


if __name__ == "__main__":
    # dataset = "Bernett_benchmarking"
    # org_ds = load_dataset(f"danliu1226/{dataset}")
    # print(org_ds)
    res_dict = torch.load("test_results.pt")
    probs, preds, labels = res_dict["probs"], res_dict["preds"], res_dict["labels"]
    metrics = compute_metrics(probs, preds, labels)
    for k, v in metrics.items():
        print(f"{k}: {v:.4f}")
