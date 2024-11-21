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


def set_seed(seed):
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


if __name__ =="__main__":
    parser=argparse.ArgumentParser()
    parser.add_argument('--batch_size',default=64,type=int,help='batch size')
    parser.add_argument('--num_chunks',default=4,type=int,help='M, namely number of micro batches')
    parser.add_argument('--seq_length', default=128, type=int, help='sequence length')
    parser.add_argument('--embedding_dim', default=4096, type=int, help='embedding dimension in a Transformer layer, 4096 for Llama-2-7b')
    parser.add_argument('--ff_dim', default=4096, type=int, help='dimension in a FeedForward layer')
    parser.add_argument('--num_iterations', default=1, type=int, help='number of iterations, namely number of batches')
    parser.add_argument('--num_stages',default=8,type=int,help='number of stages')
    parser.add_argument('--num_layers',default=32,type=int,help='number of layers')
    parser.add_argument('--num_heads',default=32,type=int,help='number of attention heads in a layer')
    parser.add_argument('--model',default='llama-2-7b',type=str,help='specify the model name')
    parser.add_argument('--dataset',default='xsum',type=str,help='specify the dataset name')

    args=parser.parse_args()

    set_seed(42)

    # device group setting
    dist.init_process_group(backend='nccl')
    world_size=dist.get_world_size()    # 程序使用的总进程数
    global_rank=dist.get_rank()
    local_size=torch.cuda.device_count()    # 当前节点上存在几张显卡
    local_rank=int(os.environ["LOCAL_RANK"]) # 进程在当前节点上的序号
    torch.cuda.set_device(local_rank%local_size) # 确保进程在多个GPU上是平分的

    '''
    Import model through hugging face. Model is on CPU by default. 
    When running this file on your machine, please annotate line62-line70 && anti-annotate line52-line60.
    '''

    # login(
    #     token="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",
    #     add_to_git_credential=True
    # )
    # model_path='meta-llama/Llama-2-7b-hf'
    # model=AutoModelForCausalLM.from_pretrained(model_path,cache_dir='transformer/model_cache')
    # tokenizer=AutoTokenizer.from_pretrained(model_path)
    # embedding_layer=model.model.embed_tokens
    # layers_list=list(model.model.layers)

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
    # # module_list=generate_module1(args)
    # model_list=copy.deepcopy(module_list)
    pipeline=Pipeline(args,module_list,world_size,global_rank,local_rank,embedding_layer,train_batches,norm_layer,lm_head)

    torch.cuda.synchronize()
    start_time=time.time()

    # pipeline execution
    for i in range(args.num_iterations):
        pipeline.optimizer.zero_grad()
        pipeline.run_pipeline(action_list)
        dist.barrier()
        torch.cuda.empty_cache()
        # 目前模型参数都在GPU里面
        pipeline.optimizer.step()
        if global_rank==0:
            print(f"--------------- finish training step {i}")
 
    torch.cuda.synchronize()
    training_time=time.time()-start_time

    verify_peak_memory(local_rank)
    if global_rank==0:
        print(torch.cuda.memory_summary(device='cuda:0', abbreviated=True))

    if global_rank==0:
        print("training time = {}".format(training_time))


    '''
    Two different method to fine-tune a complete model on only one device.
    Comparison experiment to evaulate the pipeline strategy and offload/reload strategy of Mobius, using time and memory occupation as metrics respectively.
    '''
    # if global_rank==0:
    #     print(torch.cuda.memory_summary(device='cuda:0', abbreviated=True))    
    #     another_optimizer=my_optimizer(model_list)
    #     start_time=time.time()
    #     # complete model
    #     another_device=torch.device(f'cuda:{0}')
    #     embedding_layer.to(device=another_device)
    #     norm_layer.to(device=another_device)
    #     lm_head.to(device=another_device)
    #     for it in range(args.num_iterations):
    #         another_optimizer.zero_grad()
    #         for i in range(args.num_chunks):
    #             input_data=train_batches[it*args.num_chunks+i]['input_ids']
    #             input_data=input_data.to(another_device)
    #             input_data=embedding_layer(input_data)
    #             for m in range(len(model_list)):
    #                 model_list[m]=copy.deepcopy(model_list[m]).to(another_device)
    #                 output=model_list[m](input_data)
    #                 input_data=output
    #                 input_data.requires_grad_()
    #             output=norm_layer(output)
    #             logits=lm_head(output[:, -32:, :])
    #             logits = logits.view(-1, logits.size(-1))
    #             correct_result=train_batches[it*args.num_chunks+i]["labels"].to(another_device)
    #             correct_result=correct_result.view(-1)
    #             torch.autograd.backward(F.cross_entropy(logits, correct_result))
    #             # torch.autograd.backward(output.mean())
    #         another_optimizer.step()
    #     verify_peak_memory(local_rank)
    #     print(torch.cuda.memory_summary(device='cuda:0', abbreviated=True))    
               
    #     finish_time=time.time()
    #     training_time=finish_time-start_time
    #     print("baseline training time = {}".format(training_time))


    # if global_rank==0:
    #     another_optimizer=my_optimizer(model_list)
    #     merged_model=merge_transformer_models(model_list,hidden_dim=args.embedding_dim,nhead=args.num_heads,ff_dim=args.ff_dim,dropout=0.0,norm_first=False,use_fp16=False)
    #     merged_model.to(torch.device(f'cuda:{0}'))
    #     start_time=time.time()
    #     for it in range(args.num_iterations):
    #         for i in range(args.num_chunks):
    #             pipeline.input_data[it][i].to(torch.device(f'cuda:{0}'))    
    #             pipeline.input_data[it][i].requires_grad_(True)
    #             output=merged_model(pipeline.input_data[it][i])
    #             torch.autograd.backward(output.mean())
    #             # print("continue model output",it,i,output)

    #         another_optimizer.step()
    #         # print("continue model grad",pipeline.input_data[-1].grad)
    #     finish_time=time.time()
    #     training_time=finish_time-start_time
    #     print("baseline training time = {}".format(training_time))

    
    
        
