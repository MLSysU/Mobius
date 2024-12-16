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

    # Import model through local cache files. Model is on CPU by default.
    model_path='/data/home/liuhuimin/.cache/huggingface/hub/models--meta-llama--Llama-2-7b-hf/snapshots/first_cache'
    model=LlamaForCausalLM.from_pretrained(model_path)
    config=model.config
    tokenizer=AutoTokenizer.from_pretrained(model_path)
    embedding_layer=model.model.embed_tokens
    norm_layer=model.model.norm
    lm_head=model.lm_head
    layers_list=list(model.model.layers)

    # load dataset and preprocess it.
    train_batches=None
    if global_rank==0 or global_rank==world_size-1:
        if args.dataset=='xsum':
            train_batches=preprocess_xsum(tokenizer,args.batch_size//args.num_chunks,model)

    # Generate action_list for every GPU.
    action_list=generate_action_list(world_size=world_size,num_stages=args.num_stages,num_chunks=args.num_chunks)[global_rank]
    # print('rank = '+str(global_rank)+'action_list = '+str(action_list))
    
    # Generate model shard for every stage.
    module_list=generate_module(args,config,layers_list)

    torch.cuda.synchronize()

    # 配置 profiler
    with torch.profiler.profile(
        activities=[
            torch.profiler.ProfilerActivity.CPU,
            torch.profiler.ProfilerActivity.CUDA
        ],
        schedule=torch.profiler.schedule(
            wait=2,    
            warmup=2,  # 接下来的 3 步为 warm-up
            active=1   # 随后 1 步记录 profiling 数据
        ),
        record_shapes=True,       # 记录张量形状
        with_stack=True,          # 记录调用堆栈
        on_trace_ready=torch.profiler.tensorboard_trace_handler('./test_log')  # 保存日志以供 TensorBoard 使用
    ) as prof:
        for step in range(5):
            with torch.profiler.record_function("load model"):
                model=copy.deepcopy(module_list[0]).to(f'cuda:{local_rank}')
            
            with torch.profiler.record_function("offload"):
                
            
            prof.step() 

    print(prof.key_averages().table(sort_by="cuda_memory_usage", row_limit=10))
    
    dist.destroy_process_group()