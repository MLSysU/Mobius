from transformers import AutoTokenizer, LlamaForCausalLM, DataCollatorForSeq2Seq
from datasets import load_dataset
from torch.utils.data import DataLoader


# 从缓存加载模型，默认放在CPU上
model_path='/data/home/liuhuimin/.cache/huggingface/hub/models--meta-llama--Llama-2-7b-hf/snapshots/first_cache'
model=LlamaForCausalLM.from_pretrained(model_path)
config=model.config
tokenizer=AutoTokenizer.from_pretrained(model_path)
embedding_layer=model.model.embed_tokens
layers_list=list(model.model.layers)

if tokenizer.pad_token is None:
    tokenizer.pad_token=tokenizer.eos_token

def encode_batch(batch):
    # 确保输入的 document 和 summary 字段有效
    documents = batch['document']
    summaries = batch['summary']
    
    # 检查并移除任何空字符串或非字符串内容
    documents = [doc if isinstance(doc, str) and len(doc) > 0 else " " for doc in documents]
    summaries = [summary if isinstance(summary, str) and len(summary) > 0 else " " for summary in summaries]

    # 对输入文档进行编码
    inputs = tokenizer(
        documents,
        truncation=True,
        padding="max_length",
        max_length=512,  # 根据模型最大长度设置
        return_tensors="pt"
    )

    # 对目标摘要进行编码
    targets = tokenizer(
        summaries,
        truncation=True,
        padding="max_length",
        max_length=128,  # 摘要的最大长度设置
        return_tensors="pt"
    )

    # 将编码后的输入和目标整合为一个字典
    inputs["labels"] = targets["input_ids"]
    return inputs

dataset = load_dataset("xsum", split="train", trust_remote_code=True)
encoded_dataset = dataset.map(encode_batch, batched=True)
encoded_dataset = encoded_dataset.remove_columns(["document", "summary","id"])
sample=encoded_dataset[0]
print(sample)
print("\nShapes:")
print("Input IDs shape:", len(sample["input_ids"]))
print("Attention Mask shape:", len(sample["attention_mask"]))
print("Labels shape:", len(sample["labels"]))
data_collator = DataCollatorForSeq2Seq(tokenizer=tokenizer, model=model)
batch_size = 4
train_dataloader = DataLoader(encoded_dataset, batch_size=batch_size, shuffle=True, collate_fn=data_collator)
train_batches = list(train_dataloader)

# 训练示例
for batch in train_dataloader:
    input_ids = batch["input_ids"]
    attention_mask = batch["attention_mask"]
    labels = batch["labels"]

    # 将 batch 数据转移到设备上（例如 GPU）
    # outputs = model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
    # loss = outputs.loss
    # print(f"Loss: {loss.item()}")