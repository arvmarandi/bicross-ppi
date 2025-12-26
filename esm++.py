from transformers import AutoModelForMaskedLM
import torch
from datasets import load_dataset
import os
import tqdm

from dataloader import select_subset
from utils import torch_manual_seed


def embed_sequence(model, tokenizer, sequences, max_seq_len=512, seed=42, save_path=None):
    if save_path is not None:
        os.makedirs(save_path, exist_ok=True)
        if os.path.exists(os.path.join(save_path, "logits.pth")) and os.path.exists(os.path.join(save_path, "embeddings.pth")):
            print(f"Loading existing embeddings from {save_path}")
            return torch.load(os.path.join(save_path, "logits.pth")), torch.load(os.path.join(save_path, "embeddings.pth"))

    logits_dict = {}
    embeddings_dict = {}
    for sequence in tqdm.tqdm(sequences, desc=f"Embedding sequences to {save_path}"):
        if sequence not in logits_dict:
            tokenized = tokenizer(sequence, padding=False, return_tensors='pt').to(model.device)
            output = model(**tokenized)
            logits_dict[sequence] = output.logits.detach().cpu().squeeze(0)
            embeddings_dict[sequence] = output.last_hidden_state.detach().cpu().squeeze(0)
            if logits_dict[sequence].shape[0] > max_seq_len:
                # randomly choose a contiguous segment
                with torch_manual_seed(seed):
                    start_idx = torch.randint(0, logits_dict[sequence].shape[0] - max_seq_len + 1, (1,)).item()
                logits_dict[sequence] = logits_dict[sequence][start_idx:start_idx + max_seq_len, :]
                embeddings_dict[sequence] = embeddings_dict[sequence][start_idx:start_idx + max_seq_len, :]

    if save_path:
        torch.save(logits_dict, os.path.join(save_path, "logits.pth"))
        torch.save(embeddings_dict, os.path.join(save_path, "embeddings.pth"))
    return logits_dict, embeddings_dict


if __name__ == "__main__":
    dataset = "Bernett_benchmarking"

    org_ds = load_dataset(f"danliu1226/{dataset}")
    print(org_ds)
    ds = select_subset(org_ds, 0.005, seed=0)
    print(ds)
    # print(ds['train']['query'][:5])
    
    model = AutoModelForMaskedLM.from_pretrained('Synthyra/ESMplusplus_small', trust_remote_code=True).to('cuda:2')
    tokenizer = model.tokenizer

    # sequences = ['MPRTEIN', 'MSEQWENCE']
    # tokenized = tokenizer(sequences, padding=True, return_tensors='pt')

    # # tokenized['labels'] = tokenized['input_ids'].clone() # correctly mask input_ids and set unmasked instances of labels to -100 for MLM training

    # output = model(**tokenized) # get all hidden states with output_hidden_states=True
    # print(output.logits.shape) # language modeling logits, (batch_size, seq_len, vocab_size), (2, 11, 64)
    # print(output.last_hidden_state.shape) # last hidden state of the model, (batch_size, seq_len, hidden_size), (2, 11, 960)
    # print(output.loss) # language modeling loss if you passed labels
    # #print(output.hidden_states) # all hidden states if you passed output_hidden_states=True (in tuple)

    # embedding_dict = model.embed_dataset(
    #     sequences=list(ds['train']['query']),
    #     tokenizer=model.tokenizer,
    #     batch_size=2, # adjust for your GPU memory
    #     max_len=512, # adjust for your needs
    #     full_embeddings=True, # if True, no pooling is performed
    #     embed_dtype=torch.float32, # cast to what dtype you want
    #     pooling_types=['mean', 'cls'], # more than one pooling type will be concatenated together
    #     num_workers=0, # if you have many cpu cores, we find that num_workers = 4 is fast for large datasets
    #     sql=False, # if True, embeddings will be stored in SQLite database
    #     sql_db_path='embeddings.db',
    #     save=True, # if True, embeddings will be saved as a .pth file
    #     save_path='embeddings.pth',
    # )
    # # # embedding_dict is a dictionary mapping sequences to their embeddings as tensors for .pth or numpy arrays for sql

    for mode in ['train', 'validation', 'test']:
        print(f"Embedding {mode} set...")
        p1_logits, p1_embeddings = embed_sequence(model, tokenizer, list(ds[mode]['query']), save_path=f"datasets/{dataset}/{mode}/p1")
        p2_logits, p2_embeddings = embed_sequence(model, tokenizer, list(ds[mode]['text']), save_path=f"datasets/{dataset}/{mode}/p2")
        torch.save(ds[mode]['label'], f"datasets/{dataset}/{mode}/labels.pth")

        print(f"Longest P1 sequence embedded: {max(logits.shape[0] for logits in p1_logits.values())}")
        print(f"Longest P2 sequence embedded: {max(logits.shape[0] for logits in p2_logits.values())}")