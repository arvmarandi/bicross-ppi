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


from tqdm import tqdm
import torch

def train_one_epoch_mini(
    model,
    loader,
    optimizer,
    criterion,
    max_batches=200,
    print_every=10,
):
    model.train()
    total_loss = 0.0
    loader_iter = iter(loader)

    # tqdm progress bar
    pbar = tqdm(range(max_batches), desc="Training (mini-epoch)")

    for batch_idx in pbar:
        # get next batch, reinitialize loader if needed
        try:
            p1, p1_mask, p2, p2_mask, labels = next(loader_iter)
        except StopIteration:
            loader_iter = iter(loader)
            p1, p1_mask, p2, p2_mask, labels = next(loader_iter)

        optimizer.zero_grad()
        logits = model(p1, p2, p1_mask, p2_mask)
        loss = criterion(logits, labels)

        loss.backward()
        optimizer.step()

        # accumulate loss
        total_loss += loss.item()

        # update tqdm description
        pbar.set_postfix({
            "batch_loss": f"{loss.item():.4f}",
            "avg_loss": f"{total_loss / (batch_idx+1):.4f}"
        })

        # optional console print
        if (batch_idx + 1) % print_every == 0:
            print(f"[Batch {batch_idx+1}/{max_batches}] "
                  f"Loss: {loss.item():.4f} | "
                  f"Avg: {total_loss / (batch_idx+1):.4f}")

    return total_loss / max_batches


def evaluate(model, loader, criterion):
    model.eval()
    total_loss = 0
    correct = 0
    total = 0

    with torch.no_grad():
        for i, (p1, p1_mask, p2, p2_mask, labels) in enumerate(tqdm(loader, desc="Evaluating")):
            if i >= 10:  # limit to 10 batches for faster evaluation
                break

            logits = model(p1, p2, p1_mask, p2_mask)
            loss = criterion(logits, labels)
            total_loss += loss.item()

            preds = logits.argmax(dim=1)
            correct += (preds == labels).sum().item()
            total += labels.size(0)

    acc = correct / total
    return total_loss / len(loader), acc


def predict(model, loader):
    model.eval()
    all_logits = []
    all_probs = []
    all_preds = []
    all_labels = []

    softmax = torch.nn.Softmax(dim=1)

    with torch.no_grad():
        for p1, p1_mask, p2, p2_mask, labels in loader:
            logits = model(p1, p2, p1_mask, p2_mask)
            probs = softmax(logits)[:, 1]  # probability of class 1 (interaction)

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
    device = "cuda:3"
    batch_size = 32
    emb_type = 'embeddings'  # 'embeddings' or 'logits'

    dataset_dict = load_dataset(f"danliu1226/{dataset}")
    model = AutoModelForMaskedLM.from_pretrained('Synthyra/ESMplusplus_small', trust_remote_code=True).to(device)
    tokenizer = model.tokenizer

    train_dataset = PPIDataset(dataset_dict['train'], model, tokenizer, emb_type=emb_type)
    val_dataset   = PPIDataset(dataset_dict['validation'], model, tokenizer, emb_type=emb_type)
    test_dataset  = PPIDataset(dataset_dict['test'], model, tokenizer, emb_type=emb_type)

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        collate_fn=lambda b: ppi_collate(b, device=device)
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        collate_fn=lambda b: ppi_collate(b, device=device)
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        collate_fn=lambda b: ppi_collate(b, device=device)
    )

    # ---------------------------------------------------------
    # 1. Create model (this must match your model.py)
    # ---------------------------------------------------------
    emb_dim = 960 if emb_type == 'embeddings' else 64
    hid_dim = 256
    n_layers = 4
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
    # 2. Loss and Optimizer
    # ---------------------------------------------------------
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-5, weight_decay=1e-4)


    # ---------------------------------------------------------
    # 3. Training Loop
    # ---------------------------------------------------------
    EPOCHS = 50

    # best_val_acc = 0.0
    for epoch in range(1, EPOCHS + 1):
        print(f"\n===== Epoch {epoch} =====")

        # train_loss = train_one_epoch(model, train_loader, optimizer, criterion)
        train_loss = train_one_epoch_mini(model, train_loader, optimizer, criterion, max_batches=200)
        val_loss, val_acc = evaluate(model, val_loader, criterion)

        print(f"Train Loss: {train_loss:.4f}")
        print(f"Val Loss:   {val_loss:.4f}")
        print(f"Val Acc:    {val_acc:.4f}")

        # optional: save checkpoint
        os.makedirs(f"checkpoints/{dataset}/{emb_type}", exist_ok=True)
        torch.save(model.state_dict(), f"checkpoints/{dataset}/{emb_type}/ppi_model_{epoch}.pth")

    # ---------------------------------------------------------
    # 4. Final Evaluation on Test Set
    # ---------------------------------------------------------

    print("\n===== Running Test Evaluation =====")

    model.load_state_dict(torch.load(f"checkpoints/{dataset}/{emb_type}/ppi_model.pth"))
    test_logits, test_probs, test_preds, test_labels = predict(model, test_loader)

    metrics = compute_metrics(test_probs, test_preds, test_labels)

    print("\n===== Test Metrics =====")
    for k, v in metrics.items():
        print(f"{k}: {v:.4f}")

    # optionally save results
    # torch.save({
    #     "logits": test_logits,
    #     "probs": test_probs,
    #     "preds": test_preds,
    #     "labels": test_labels,
    #     "metrics": metrics,
    # }, "test_results.pt")


if __name__ == "__main__":
    main()
