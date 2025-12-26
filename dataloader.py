import torch
import torch.nn.functional as F
from datasets import DatasetDict, load_dataset

from utils import torch_manual_seed


class PPIDataset(torch.utils.data.Dataset):
    def __init__(self, dataset, esm_model, esm_tokenizer, emb_type='embeddings', max_seq_len=512, seed=42):
        """
        p1_embeddings: list of tensors, shape [L1_i, emb_dim]
        p2_embeddings: list of tensors, shape [L2_i, emb_dim]
        labels:        list/array of ints
        """
        assert emb_type in ['embeddings', 'logits'], "emb_type must be 'embeddings' or 'logits'"

        self.p1_sequences = list(dataset['query'])
        self.p2_sequences = list(dataset['text'])
        self.labels = list(dataset['label'])
        self.esm_model = esm_model
        self.esm_tokenizer = esm_tokenizer
        self.emb_type = emb_type
        self.max_seq_len = max_seq_len
        self.seed = seed

    def get_esm_embeddings(self, idx):
        p1_seq = self.p1_sequences[idx]
        p2_seq = self.p2_sequences[idx]
        p1_tokenized = self.esm_tokenizer(p1_seq, padding=False, return_tensors='pt').to(self.esm_model.device)
        p2_tokenized = self.esm_tokenizer(p2_seq, padding=False, return_tensors='pt').to(self.esm_model.device)
        output1 = self.esm_model(**p1_tokenized)
        output2 = self.esm_model(**p2_tokenized)
        if self.emb_type == 'embeddings':
            p1_emb = output1.last_hidden_state.detach().cpu().squeeze(0)
            p2_emb = output2.last_hidden_state.detach().cpu().squeeze(0)
        else:  # logits
            p1_emb = output1.logits.detach().cpu().squeeze(0)
            p2_emb = output2.logits.detach().cpu().squeeze(0)
        
        if p1_emb.shape[0] > self.max_seq_len:
            with torch_manual_seed(self.seed):
                start_idx = torch.randint(0, p1_emb.shape[0] - self.max_seq_len + 1, (1,)).item()
            p1_emb = p1_emb[start_idx:start_idx + self.max_seq_len, :]
        if p2_emb.shape[0] > self.max_seq_len:
            with torch_manual_seed(self.seed):
                start_idx = torch.randint(0, p2_emb.shape[0] - self.max_seq_len + 1, (1,)).item()
            p2_emb = p2_emb[start_idx:start_idx + self.max_seq_len, :]

        return p1_emb, p2_emb

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        p1_emb, p2_emb = self.get_esm_embeddings(idx)
        return {
            "p1": p1_emb,        # (L1_i, emb_dim)
            "p2": p2_emb,        # (L2_i, emb_dim)
            "label": int(self.labels[idx])
        }


def pad_and_mask(emb_list, device="cuda"):
    """
    emb_list: list of tensors, each shape (L_i, D)
    returns:
        padded: (B, L_max, D)
        mask:   (B, L_max) with 1=valid, 0=pad
    """
    lengths = [emb.shape[0] for emb in emb_list]
    max_len = max(lengths)
    emb_dim = emb_list[0].shape[1]

    padded = torch.zeros(len(emb_list), max_len, emb_dim, device=device)
    mask   = torch.zeros(len(emb_list), max_len, device=device)

    for i, emb in enumerate(emb_list):
        L = emb.shape[0]
        padded[i, :L] = emb.to(device)
        mask[i, :L] = 1

    return padded, mask


def ppi_collate(batch, device="cuda"):
    """
    batch: list of dicts:
        {
            'p1': tensor(L1, D),
            'p2': tensor(L2, D),
            'label': int
        }

    Returns:
        p1_padded, p1_mask,
        p2_padded, p2_mask,
        labels
    """

    p1_list = [sample["p1"] for sample in batch]
    p2_list = [sample["p2"] for sample in batch]
    labels  = torch.tensor([sample["label"] for sample in batch], device=device)

    # Use the pad_and_mask function defined earlier
    p1_padded, p1_mask = pad_and_mask(p1_list, device=device)
    p2_padded, p2_mask = pad_and_mask(p2_list, device=device)

    p1_attn_mask = p1_mask.unsqueeze(1).unsqueeze(2)  # (B, 1, 1, L1)
    p2_attn_mask = p2_mask.unsqueeze(1).unsqueeze(2)  # (B, 1, 1, L2)

    return p1_padded, p1_attn_mask, p2_padded, p2_attn_mask, labels


def select_subset(ds: DatasetDict, ratio: float, seed: int = 42):
    """
    Select a random subset of each split in a DatasetDict.

    Args:
        ds (DatasetDict): a HF DatasetDict with keys like "train", "validation", "test".
        ratio (float): fraction of each split to keep (0 < ratio <= 1).
        seed (int): random seed for reproducibility.

    Returns:
        DatasetDict: new DatasetDict with subsampled splits.
    """
    if not isinstance(ds, DatasetDict):
        raise ValueError("Input must be a HuggingFace DatasetDict.")

    if not (0 < ratio <= 1):
        raise ValueError("ratio must be between 0 and 1.")

    new_splits = {}
    for split_name, split_data in ds.items():
        # Shuffle the split so selection is random
        split_data = split_data.shuffle(seed=seed)

        # Compute target subset size
        subset_size = int(len(split_data) * ratio)

        # Select the first subset_size rows
        new_splits[split_name] = split_data.select(range(subset_size))

    return DatasetDict(new_splits)


def load_data(dataset, mode, emb_type='embeddings', ratio=0.001, seed=0):
    assert mode in ['train', 'validation', 'test'], "mode must be 'train', 'validation', or 'test'"
    assert emb_type in ['embeddings', 'logits'], "emb_type must be 'embeddings' or 'logits'"

    print(f"Loading {mode} set from {dataset} with ratio {ratio} and seed {seed}...")
    org_ds = load_dataset(f"danliu1226/{dataset}")
    ds = select_subset(org_ds, ratio, seed=seed)

    p1_embedding_dict = torch.load(f"datasets/{dataset}/{mode}/p1/{emb_type}.pth")
    p2_embedding_dict = torch.load(f"datasets/{dataset}/{mode}/p2/{emb_type}.pth")
    labels = torch.load(f"datasets/{dataset}/{mode}/labels.pth")
    p1_embeddings = [p1_embedding_dict[seq] for seq in ds[mode]['query']]
    p2_embeddings = [p2_embedding_dict[seq] for seq in ds[mode]['text']]
    print(f"Loaded {len(p1_embeddings)} P1 embeddings and {len(p2_embeddings)} P2 embeddings.")

    return p1_embeddings, p2_embeddings, labels


if __name__ == "__main__":
    dataset = "cross_species_benchmarking"
    mode = "train"

    org_ds = load_dataset(f"danliu1226/{dataset}")
    ds = select_subset(org_ds, 0.001, seed=0)
    print(ds)

    p1_embedding_dict = torch.load(f"datasets/{dataset}/{mode}/p1/embeddings.pth")
    p2_embedding_dict = torch.load(f"datasets/{dataset}/{mode}/p2/embeddings.pth")
    labels = torch.load(f"datasets/{dataset}/{mode}/labels.pth")
    p1_embeddings = [p1_embedding_dict[seq] for seq in ds[mode]['query']]
    p2_embeddings = [p2_embedding_dict[seq] for seq in ds[mode]['text']]
    print(f"Loaded {len(p1_embeddings)} P1 embeddings and {len(p2_embeddings)} P2 embeddings.")

    ppi_dataset = PPIDataset(p1_embeddings, p2_embeddings, labels)
    dataloader = torch.utils.data.DataLoader(ppi_dataset, batch_size=16, shuffle=True,
                                             collate_fn=lambda x: ppi_collate(x, device="cpu"))
    
    for batch in dataloader:
        p1_padded, p1_mask, p2_padded, p2_mask, labels = batch
        print("P1 padded:", p1_padded.shape)
        print("P1 mask:", p1_mask.shape)
        print("P2 padded:", p2_padded.shape)
        print("P2 mask:", p2_mask.shape)
        print("Labels:", labels.shape)
        # break

