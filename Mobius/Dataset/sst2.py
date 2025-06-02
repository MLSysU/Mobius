import torch
from datasets import load_dataset
from torch.utils.data import DataLoader

def preprocess_sst2(tokenizer,batch_size):
    sst2_dataset = load_dataset("glue", "sst2")
    def tokenize_function(examples):
        if tokenizer.pad_token is None:
            tokenizer.pad_token=tokenizer.eos_token
        return tokenizer(examples['sentence'],truncation=True,padding='max_length',max_length=args.seq_length)
    tokenized_datasets=sst2_dataset.map(tokenize_function,batched=True)
    tokenized_datasets = tokenized_datasets.remove_columns(["sentence", "idx"])
    tokenized_datasets = tokenized_datasets.rename_column("label", "labels")
    tokenized_datasets.set_format("torch")
    train_dataloader = DataLoader(tokenized_datasets["train"], batch_size=batch_size, shuffle=True)
    train_batches = list(train_dataloader)

    return train_batches
