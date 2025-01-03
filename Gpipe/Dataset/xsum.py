from transformers import DataCollatorForSeq2Seq
from datasets import load_dataset
from torch.utils.data import DataLoader


def preprocess_xsum(tokenizer,batch_size,model):
    if tokenizer.pad_token is None:
        tokenizer.pad_token=tokenizer.eos_token

    def encode_batch(batch):
        documents = batch['document']
        summaries = batch['summary']
        
        # 检查并移除任何空字符串或非字符串内容
        documents = [doc if isinstance(doc, str) and len(doc) > 0 else " " for doc in documents]
        summaries = [summary if isinstance(summary, str) and len(summary) > 0 else " " for summary in summaries]

        inputs = tokenizer(
            documents,
            truncation=True,
            padding="max_length",
            max_length=256,  
            return_tensors="pt"
        )

        targets = tokenizer(
            summaries,
            truncation=True,
            padding="max_length",
            max_length=64,  
            return_tensors="pt"
        )

        inputs["labels"] = targets["input_ids"]
        return inputs

    dataset = load_dataset("xsum", split="train", trust_remote_code=True)
    encoded_dataset = dataset.map(encode_batch, batched=True)
    encoded_dataset = encoded_dataset.remove_columns(["document", "summary","id"])
    data_collator = DataCollatorForSeq2Seq(tokenizer=tokenizer, model=model)
    train_dataloader = DataLoader(encoded_dataset, batch_size=batch_size, shuffle=False, collate_fn=data_collator)
    train_batches = list(train_dataloader)

    return train_batches