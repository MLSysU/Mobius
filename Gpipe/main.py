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



def set_seed(seed):
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


if __name__ =="__main__":
    parser=argparse.ArgumentParser()
    parser.add_argument('--batch_size',default=64,type=int,help='batch size')
    parser.add_argument('--num_chunks',default=4,type=int,help='M, namely number of micro batches')
    parser.add_argument('--seq_length', default=128, type=int, help='sequence length, should not be changed')
    parser.add_argument('--embedding_dim', default=4096, type=int, help='embedding dimension in a Transformer layer, 4096 for Llama-2-7b')
    parser.add_argument('--ff_dim', default=4096, type=int, help='dimension in a FeedForward layer')
    parser.add_argument('--num_iterations', default=2, type=int, help='number of iterations, namely number of batches')
    parser.add_argument('--num_stages',default=8,type=int,help='number of stages')
    parser.add_argument('--num_layers',default=32,type=int,help='number of layers')
    parser.add_argument('--num_heads',default=32,type=int,help='number of attention heads in a layer')
    parser.add_argument('--model',default='llama-2-7b',type=str,help='specify the model name')
    parser.add_argument('--dataset',default='xsum',type=str,help='specify the dataset name')
    parser.add_argument('--save_results',default='test_result.txt',type=str,help='file to save the results')
    parser.add_argument('--use_prefetch', action='store_true', help='Enable prefetch trick')
    parser.add_argument('--no_prefetch', action='store_false', dest='use_prefetch', help='Disable prefetch trick')
    parser.add_argument('--use_offload', action='store_true',help='use model offload strategy in forward process')
    parser.add_argument('--no_offload',action='store_false',dest='use_offload',help='Disable model offload strategy')

    args=parser.parse_args()

    set_seed(42)



    # device group setting
    dist.init_process_group(backend='nccl')
    world_size=dist.get_world_size()    # 程序使用的总进程数
    global_rank=dist.get_rank()
    local_size=torch.cuda.device_count()    # 当前节点上存在几张显卡
    local_rank=int(os.environ["LOCAL_RANK"]) # 进程在当前节点上的序号
    torch.cuda.set_device(local_rank%local_size) # 确保进程在多个GPU上是平分的

    if global_rank==0:
        with open(args.save_results,'a') as f:
            print("num_iterations = {}".format(args.num_iterations),file=f)
            print("batch_size = {}".format(args.batch_size),file=f)
            print("num_layers = {}".format(args.num_layers),file=f)
            print("num_stages = {}".format(args.num_stages),file=f)
            print("use_prefetch = {}".format(args.use_prefetch),file=f)
            print("use_offload = {}".format(args.use_offload),file=f)

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

    # module_list=generate_module1(args)
    # model_list=copy.deepcopy(module_list)

    # sustain a prefetch thread and a offload thread for every GPU.
    '''
    PrefetchThreadManager=ThreadManager()
    '''
    OffloadThreadManager=ThreadManager()
    '''
    pipeline=Pipeline(args,module_list,world_size,global_rank,local_rank,embedding_layer,train_batches,norm_layer,lm_head,PrefetchThreadManager,OffloadThreadManager)
    '''
    pipeline=Pipeline(args,module_list,world_size,global_rank,local_rank,embedding_layer,train_batches,norm_layer,lm_head,OffloadThreadManager)

    torch.cuda.synchronize()
    
    

    # 配置 profiler
    with torch.profiler.profile(
        activities=[
            torch.profiler.ProfilerActivity.CPU,
            torch.profiler.ProfilerActivity.CUDA
        ],
        schedule=torch.profiler.schedule(  
            wait=2, 
            warmup=2,  # 接下来的 2 步为 warm-up
            active=1   # 随后 1 步记录 profiling 数据
        ),
        record_shapes=True,       # 记录张量形状
        with_stack=True,          # 记录调用堆栈
        on_trace_ready=torch.profiler.tensorboard_trace_handler('./zero_log')  # 保存日志以供 TensorBoard 使用
    ) as prof:
        for step in range(5):   
            if step==4:
                num_iterations=args.num_iterations
            else:
                num_iterations=2
            # pipeline execution
            training_time=0
            for i in range(num_iterations):
                start_time=time.time()
                pipeline.optimizer.zero_grad()
                pipeline.run_pipeline(action_list)
                dist.barrier()
                OffloadThreadManager.wait_for_task_completion()
                torch.cuda.synchronize()
                end_time=time.time()
                start_step_time=time.time()
                for param in module_list[0].parameters():
                    if param.grad is not None:
                        print(param.grad)
                pipeline.optimizer.step()
                end_step_time=time.time()
                # torch.cuda.empty_cache()
                if global_rank==0:
                    training_time+=end_time-start_time
                    with open(args.save_results,'a') as f:
                        print("step time = {}".format(end_step_time-start_step_time),file=f)
                        print(f"--------------- finish training step {i}",file=f)
                        print(i, time.time()-start_time,file=f) 
            torch.cuda.empty_cache()
            '''
            pipeline.PrefetchThreadManager.shutdown()
            '''
            pipeline.OffloadThreadManager.shutdown()
            '''
            PrefetchThreadManager=ThreadManager()
            '''
            OffloadThreadManager=ThreadManager()
            '''
            pipeline=Pipeline(args,module_list,world_size,global_rank,local_rank,embedding_layer,train_batches,norm_layer,lm_head,PrefetchThreadManager,OffloadThreadManager) 
            '''
            pipeline=Pipeline(args,module_list,world_size,global_rank,local_rank,embedding_layer,train_batches,norm_layer,lm_head,OffloadThreadManager)
            prof.step() 

    print(prof.key_averages().table(sort_by="cuda_memory_usage", row_limit=10))

    with open(args.save_results,'a') as f:
        verify_peak_memory(local_rank,f)
    if global_rank==0:
        with open(args.save_results,'a') as f:
            print(torch.cuda.memory_summary(device='cuda:0', abbreviated=True),file=f)

    if global_rank==0:
        with open(args.save_results,'a') as f:
            print("training time = {}".format(training_time),file=f)
    
    dist.destroy_process_group()
    '''

    Two different method to fine-tune a complete model on only one device.
    Comparison experiment to evaulate the pipeline strategy and offload/reload strategy of Mobius, using time and memory occupation as metrics respectively.
    '''
    '''
    if global_rank==0:
        another_optimizer=my_optimizer(model_list)
        start_time=time.time()
        # complete model
        another_device=torch.device(f'cuda:{0}')
        embedding_layer.half().to(device=another_device)
        norm_layer.half().to(device=another_device)
        lm_head.half().to(device=another_device)
        for it in range(args.num_iterations):
            my_models=[]
            another_optimizer.zero_grad()
            for i in range(args.num_chunks):
                input_data=train_batches[it*args.num_chunks+i]['input_ids']
                input_data=input_data.to(another_device)
                input_data=embedding_layer(input_data)      
                for m in range(len(model_list)):
                    input_data.requires_grad_()
                    input_data.retain_grad()
                    if m==1:
                        first_data=input_data
                    if len(my_models)>m:
                        my_model=my_models[m]
                    else:    
                        my_model=copy.deepcopy(model_list[m]).half().to(another_device)
                        my_models.append(my_model)
                    output=my_model(input_data)
                    input_data=output
                    
                output=norm_layer(output)
                logits=lm_head(output[:, -32:, :])
                logits = logits.view(-1, logits.size(-1))
                correct_result=train_batches[it*args.num_chunks+i]["labels"].to(another_device)
                correct_result=correct_result.view(-1)
                torch.autograd.backward(F.cross_entropy(logits, correct_result))

                if i==0:
                    print("continue input grad",it,i,first_data.grad)
                if i==0:
                    print("continue input grad",it,i,first_data.grad)
                if i==0:
                    print("continue model output",it,i,output)
            for i in range(len(my_models)):
                for name,param in my_models[i].named_parameters():
                    if param.grad is not None:
                        model_param=dict(model_list[i].named_parameters())[name]
                        double_grad=param.grad.to(dtype=torch.float32,device='cpu')
                        model_param.grad=torch.empty_like(double_grad,device='cpu').copy_(double_grad)
            # for name,param in model_list[0].named_parameters():
            #     if param.grad is not None:
            #         print("continue model",it,name,param)
            #         print("continue model grad",it,name,param.grad)
            start_step_time=time.time()
            another_optimizer.step()
            end_step_time=time.time()
            print("step time = {}".format(end_step_time-start_step_time))
            # for name, param in model_list[1].named_parameters():
            #     print("continue",it,name,param)

        verify_peak_memory(local_rank,None)
        print(torch.cuda.memory_summary(device='cuda:0', abbreviated=True))    
               
        finish_time=time.time()
        training_time=finish_time-start_time
        print("baseline training time = {}".format(training_time))
    
    dist.destroy_process_group()
    '''

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

    
    
        
