import torch.nn.functional as F 
import torch 
from torch.utils.data import DataLoader, Dataset
import numpy as np
from collections import Counter

class Encoder(object):
    """convert between strings and their one-hot representations"""
    def __init__(self, alphabet: str = 'ARNDCQEGHILKMFPSTWYV'):
        self.alphabet = alphabet
        self.a_to_t = {a: i for i, a in enumerate(self.alphabet)}
        self.t_to_a = {i: a for i, a in enumerate(self.alphabet)}

    @property
    def vocab_size(self) -> int:
        return len(self.alphabet)
    
    @property
    def vocab(self) -> np.ndarray:
        return np.array(list(self.alphabet))
    
    @property
    def tokenized_vocab(self) -> np.ndarray:
        return np.array([self.a_to_t[a] for a in self.alphabet])

    def onehotize(self, batch):
        #create a tensor, and then onehotize using scatter_
        onehot = torch.zeros(len(batch), self.vocab_size)
        onehot.scatter_(1, batch.unsqueeze(1), 1)
        return onehot
    
    def encode(self, seq_or_batch: str or list, return_tensor = True) -> np.ndarray or torch.Tensor:
        if isinstance(seq_or_batch, str):
            encoded_list = [self.a_to_t[a] for a in seq_or_batch]
        else:
            encoded_list = [[self.a_to_t[a] for a in seq] for seq in seq_or_batch]
        return torch.tensor(encoded_list) if return_tensor else encoded_list
    
    def decode(self, x: np.ndarray or list or torch.Tensor) -> str or list:
        if isinstance(x, np.ndarray):
            x = x.tolist()
        elif isinstance(x, torch.Tensor):
            x = x.tolist()

        if isinstance(x[0], list):
            return [''.join([self.t_to_a[t] for t in xi]) for xi in x]
        else:
            return ''.join([self.t_to_a[t] for t in x])

def check_duplicates(sequences):
    """
    Check if there are duplicates in the list of sequences.
    """
    sequence_counts = Counter(sequences)
    return list(sequence_counts.values()), list(sequence_counts.keys())

class ProteinDataset(Dataset):
    def __init__(self, df, scenario, task, tokenizer, seq_len):
        df = df.sort_values(by='score', ascending=False)

        self.df = df 
        if scenario == "gfp":
            self.DMS_score_min, self.DMS_score_max = 1.2834, 4.1231
        elif scenario == "aav": 
            self.DMS_score_min, self.DMS_score_max = 0.0, 19.5365

        self.tokenizer = tokenizer
        self.seq_len = seq_len

    def __len__(self):
        return len(self.df)
    
    def normalize_fitness(self, score, min_val, max_val):
        # normalize fitness values to [0,1]
        return (score - min_val) / (max_val - min_val)

    def filter_tokens(self, embeddings):
        # given embeddings of (1,seq_len,33) -> (1,seq_len,24)
        embeddings = embeddings[:, :, self.valid_ids]
        return embeddings 

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        input_seq = self.tokenizer.encode(row['sequence'])
        if input_seq.shape[0] == 237:
            input_seq = F.pad(input_seq, (0, 3), "constant", 0)

        # one hot encoding 
        input_seq = F.one_hot(input_seq, num_classes=20).float()

        # Add the label to the inputs dictionary
        DMS_score = self.normalize_fitness(row['score'], self.DMS_score_min, self.DMS_score_max) 
        DMS_score = torch.tensor(DMS_score, dtype=torch.float32)
        
        return input_seq, DMS_score

class ProteinDatasetOrig(Dataset):
    def __init__(self, df, scenario, task, tokenizer, seq_len):
        df = df.sort_values(by='score', ascending=False)

        self.df = df 
        if scenario == "gfp":
            self.DMS_score_min, self.DMS_score_max = 1.2834, 4.1231
        elif scenario == "aav": 
            self.DMS_score_min, self.DMS_score_max = 0.0, 19.5365

        self.tokenizer = tokenizer
        self.seq_len = seq_len

    def __len__(self):
        return len(self.df)
    
    def normalize_fitness(self, score, min_val, max_val):
        # normalize fitness values to [0,1]
        return (score - min_val) / (max_val - min_val)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        input_seq = self.tokenizer.encode(row['sequence'])
        if input_seq.shape[0] == 237:
            input_seq = F.pad(input_seq, (0, 3), "constant", 0)

        # Add the label to the inputs dictionary
        DMS_score = self.normalize_fitness(row['score'], self.DMS_score_min, self.DMS_score_max) 
        DMS_score = torch.tensor(DMS_score, dtype=torch.float32)
        
        return input_seq, DMS_score