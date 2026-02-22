from torch.utils.data import Dataset
import torch
import os
import random
from datasets import load_dataset

class PretrainDataset(Dataset):
    def __init__(self, data_path, tokenizer, max_length = 512):
        super().__init__()
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.sample = load_dataset("json", data_files = data_path, split = "train")

    def __len__(self):
        return len(self.sample)
    
    def __getitem__(self, index):
        self.sample = self.sample[index]
        tokens = self.tokenizer(str(self.sample["text"]), add_special_tokens = False, max_length = self.max_length - 2, truncation = True).input_ids
        tokens = [self.tokenizer.bos_token_id] + tokens [self.tokenizer.eos_token_id]
        input_ids = tokens + [self.tokenizer.pad_token_id] * (self.max_length - len(tokens))
        labels = input_ids.clone()
        labels[input_ids == self.tokenizer.pad_token_id] = -100
        return input_ids, labels
    
