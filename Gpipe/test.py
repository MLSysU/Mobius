import argparse
from Runtime import *
from Dataset import *
import torch
import torch.distributed as dist
import torch.nn.functional as F
import os
from utils import *
import time
import copy
from transformers import AutoModelForCausalLM,AutoTokenizer,DataCollatorForSeq2Seq
from transformers import LlamaTokenizer,LlamaForCausalLM
from huggingface_hub import login
from datasets import load_dataset
from torch.utils.data import DataLoader
import torch.profiler


parser=argparse.ArgumentParser()
parser.add_argument('--num_stages',default=8,type=int,help='number of stages')
args=parser.parse_args()

# device group setting
dist.init_process_group(backend='nccl')
world_size=dist.get_world_size()    # 程序使用的总进程数
global_rank=dist.get_rank()
local_size=torch.cuda.device_count()    # 当前节点上存在几张显卡
local_rank=int(os.environ["LOCAL_RANK"]) # 进程在当前节点上的序号
torch.cuda.set_device(local_rank%local_size) # 确保进程在多个GPU上是平分的

# 模拟一个模型
model_path='/data/home/liuhuimin/.cache/huggingface/hub/models--meta-llama--Llama-2-7b-hf/snapshots/first_cache'
model=LlamaForCausalLM.from_pretrained(model_path)
config=model.config
tokenizer=AutoTokenizer.from_pretrained(model_path)
embedding_layer=model.model.embed_tokens
norm_layer=model.model.norm
lm_head=model.lm_head
layers_list=list(model.model.layers)
module_list=generate_module(args,config,layers_list)

my_stage_id = 0
device = torch.device(f"cuda:{global_rank}")



# 深拷贝和传输到 GPU
module = copy.deepcopy(module_list[my_stage_id]).half()
# 测量 PCIe 数据传输时间
start_time = time.time()
module.to(device, non_blocking=True)

end_time = time.time()

# 计算模型的总大小
total_params = sum(p.numel() for p in module.parameters())
data_size_bytes = total_params * 2  # FP16 每个参数占用 2 字节

# 计算带宽 (单位: GB/s)
elapsed_time = end_time - start_time
pcie_bandwidth = (data_size_bytes / elapsed_time) / 1e9

print(f"数据量: {data_size_bytes / 1e6:.2f} MB")
print(f"耗时: {elapsed_time:.6f} 秒")
print(f"估算 PCIe 带宽: {pcie_bandwidth:.2f} GB/s")

data=torch.rand(16,128,4096).to(device)
if global_rank==0:
    target_rank=1
if global_rank==1:
    target_rank=0

start_time = time.time()
if global_rank==0:
    send=dist.isend(tensor=data,dst=target_rank)
if global_rank==1:
    recv=dist.irecv(tensor=data,src=0)
end_time = time.time()

# 计算带宽 (单位: GB/s)
elapsed_time = end_time - start_time
data_size_bytes=16*128*4096*4
pcie_bandwidth = (data_size_bytes / elapsed_time) / 1e9

print(f"数据量: {data_size_bytes / 1e6:.2f} MB")
print(f"耗时: {elapsed_time:.6f} 秒")
print(f"估算 PCIe 带宽: {pcie_bandwidth:.2f} GB/s")
