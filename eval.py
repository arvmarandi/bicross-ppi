import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from transformers import AutoModelForMaskedLM
from datasets import load_dataset
from tqdm import tqdm
import os
from sklearn.metrics import (
    roc_auc_score,
    average_precision_score,
    precision_score,
    recall_score,
    f1_score
)

# -------------------------------------------------------------
# import your dataset, collate function, decoder & predictor
# -------------------------------------------------------------
from dataloader import *
from model import Decoder, DecoderLayer, PPIPredictor, SelfAttention, PositionwiseFeedforward # your decoder + predictor


# =============================================================
# ======================= MAIN TRAINING ========================
# =============================================================

def train_one_epoch(model, loader, optimizer, criterion):
    model.train()
    total_loss = 0

    for p1, p1_mask, p2, p2_mask, labels in tqdm(loader, desc="Training"):
        optimizer.zero_grad()

        logits = model(p1, p2, p1_mask, p2_mask)  # forward pass
        loss = criterion(logits, labels)

        loss.backward()
        optimizer.step()

        total_loss += loss.item()

    return total_loss / len(loader)


def evaluate(model, loader, criterion):
    model.eval()
    total_loss = 0
    correct = 0
    total = 0

    with torch.no_grad():
        for p1, p1_mask, p2, p2_mask, labels in tqdm(loader, desc="Evaluating"):
            logits = model(p1, p2, p1_mask, p2_mask)
            loss = criterion(logits, labels)
            total_loss += loss.item()

            preds = logits.argmax(dim=1)
            correct += (preds == labels).sum().item()
            total += labels.size(0)

    acc = correct / total
    return total_loss / len(loader), acc


def predict(model, loader, max_batches=None):
    model.eval()
    all_logits = []
    all_probs = []
    all_preds = []
    all_labels = []

    softmax = torch.nn.Softmax(dim=1)

    # If max_batches is None → use full loader
    # Else → use min(max_batches, len(loader)) for progress bar length
    total_batches = max_batches if max_batches is not None else len(loader)

    with torch.no_grad():
        # wrap loader with tqdm directly
        for i, (p1, p1_mask, p2, p2_mask, labels) in tqdm(
            enumerate(loader),
            total=total_batches,
            desc="Predicting",
        ):
            if max_batches is not None and i >= max_batches:
                break

            logits = model(p1, p2, p1_mask, p2_mask)
            probs = softmax(logits)[:, 1]
            preds = logits.argmax(dim=1)

            all_logits.append(logits.cpu())
            all_probs.append(probs.cpu())
            all_preds.append(preds.cpu())
            all_labels.append(labels.cpu())

    return (
        torch.cat(all_logits, dim=0),
        torch.cat(all_probs, dim=0),
        torch.cat(all_preds, dim=0),
        torch.cat(all_labels, dim=0),
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


def main():

    # ---------------------------------------------------------
    # 0. Load your embedding lists (example)
    # ---------------------------------------------------------
    # p1_train, p2_train, labels_train = ...
    # p1_val, p2_val, labels_val = ...

    dataset = "Bernett_benchmarking"
    device = "cuda:5"
    batch_size = 32
    emb_type = 'embeddings'  # 'embeddings' or 'logits'

    dataset_dict = load_dataset(f"danliu1226/{dataset}")
    model = AutoModelForMaskedLM.from_pretrained('Synthyra/ESMplusplus_small', trust_remote_code=True).to(device)
    tokenizer = model.tokenizer
    test_dataset  = PPIDataset(dataset_dict['test'], model, tokenizer, emb_type=emb_type)

    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=True,
        collate_fn=lambda b: ppi_collate(b, device=device)
    )

    # ---------------------------------------------------------
    # 1. Create model (this must match your model.py)
    # ---------------------------------------------------------
    emb_dim = 960       # ESM++ embedding dimension
    hid_dim = 256
    n_layers = 2
    n_heads = 8
    pf_dim = 512
    dropout = 0.1

    decoder = Decoder(
        emb_dim, hid_dim, n_layers, n_heads, pf_dim,
        decoder_layer=DecoderLayer,
        self_attention=SelfAttention,
        positionwise_feedforward=PositionwiseFeedforward,
        dropout=dropout,
        device=device
    ).to(device)

    model = PPIPredictor(decoder, device).to(device)

    # ---------------------------------------------------------
    # 4. Final Evaluation on Test Set
    # ---------------------------------------------------------

    print("\n===== Running Test Evaluation =====")

    iteration=1
    model.load_state_dict(torch.load(f"checkpoints/{dataset}/{emb_type}/ppi_model_2.pth"))
    test_logits, test_probs, test_preds, test_labels = predict(model, test_loader, max_batches=50)

    metrics = compute_metrics(test_probs, test_preds, test_labels)

    print("\n===== Test Metrics =====")
    for k, v in metrics.items():
        print(f"{k}: {v:.4f}")

    # optionally save results
    torch.save({
        "logits": test_logits,
        "probs": test_probs,
        "preds": test_preds,
        "labels": test_labels,
        "metrics": metrics,
    }, "test_results.pt")


if __name__ == "__main__":
    main()
