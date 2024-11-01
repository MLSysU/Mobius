import ollama
import torch
from torch import nn
from transformers import LlamaTokenizer
from transformers import AutoModelForCausalLM,AutoTokenizer 
from huggingface_hub import login,list_repo_files
from datasets import load_dataset
import psutil
from transformers import LlamaForCausalLM
from torch.utils.data import DataLoader

class SequentialWithCustomForward(nn.Sequential):
    def forward(self, input_tensor):
        hidden_states = input_tensor 
        i=0 
        for layer in self:
            output = layer(hidden_states)  # Get the output from the layer
            if isinstance(output, tuple):
                hidden_states = output[0]  # Use the first item if it's a tuple
            else:
                hidden_states = output  # Otherwise, it's a single tensor
        return hidden_states

login(
    token="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",
    add_to_git_credential=True
)
# model_path='openlm-research/open_llama_7b_v2'
# tokenizer=LlamaTokenizer.from_pretrained(model_path)
# model=LlamaForCausalLM.from_pretrained(model_path,torch_dtype=torch.float16,device_map='auto')


model_path='/data/home/liuhuimin/.cache/huggingface/hub/models--meta-llama--Llama-2-7b-hf/snapshots/first_cache'
model=LlamaForCausalLM.from_pretrained(model_path)
embedding_layer=model.model.embed_tokens           
tokenizer=AutoTokenizer.from_pretrained(model_path)
'''
加载checkpoint的参数
custom_state_dict = torch.load('my_model.pt', map_location=torch.device('cpu'))
model.load_state_dict(custom_state_dict)
'''

def tokenize_function(examples):
    if tokenizer.pad_token is None:
        tokenizer.pad_token=tokenizer.eos_token
    return tokenizer(examples['sentence'],truncation=True,padding='max_length',max_length=128)
sst2_dataset=load_dataset("glue","sst2")
tokenized_datasets=sst2_dataset.map(tokenize_function,batched=True)
tokenized_datasets = tokenized_datasets.remove_columns(["sentence", "idx"])
tokenized_datasets = tokenized_datasets.rename_column("label", "labels")
tokenized_datasets.set_format("torch")
train_dataloader = DataLoader(tokenized_datasets["train"], batch_size=16, shuffle=True)
train_batches = list(train_dataloader)
print("dataloader",len(train_batches))
first_batch=train_batches[0]
first_ids=first_batch['input_ids']
embedding_ids=embedding_layer(first_ids)
print(embedding_ids.shape)
                                                                                                                                                                                                                                                                                                                                                                                                                                                                                       
print(first_ids.shape)
                                      
                                                                                    
                                 

# 统计模型参数量                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                             
total_params = sum(p.numel() for p in model.parameters())
print(f"模型的总参数量: {total_params / 1e9:.2f} B")  # 将参数量以 "B"（billion）为单位输出


layers_list=list(model.model.layers)

input_tensor=torch.rand(64,128,4096)
stage_layers=layers_list[0:2]
stage_model = SequentialWithCustomForward(*stage_layers)
activation=stage_model(input_tensor)




