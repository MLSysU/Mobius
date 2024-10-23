import argparse
from Runtime import *
import torch
import torch.distributed as dist
import os
from utils import *
import time
import copy


def set_seed(seed):
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

def my_optimizer(model_list):
    parameters=[]
    for module in model_list:
        module_parameter=list(module.parameters())  # parameters()是nn.Module自带的函数，返回一个生成器，可以迭代调用出模型每一层的参数大小
        parameters+=module_parameter
    return torch.optim.Adam(parameters,lr=0.0001,weight_decay=1e-3)



if __name__ =="__main__":
    parser=argparse.ArgumentParser()
    parser.add_argument('--batch_size',default=64,type=int,help='batch size')
    parser.add_argument('--num_chunks',default=4,type=int,help='M, namely number of micro batches')
    parser.add_argument('--seq_length', default=128, type=int, help='sequence length')
    parser.add_argument('--embedding_dim', default=1024, type=int, help='embedding dimension in a Transformer layer')
    parser.add_argument('--ff_dim', default=4096, type=int, help='dimension in a FeedForward layer')
    parser.add_argument('--num_iterations', default=2, type=int, help='number of iterations, namely number of batches')
    parser.add_argument('--num_stages',default=8,type=int,help='number of stages')
    parser.add_argument('--num_layers',default=24,type=int,help='number of layers')
    parser.add_argument('--num_heads',default=32,type=int,help='number of attention heads in a layer')



    args=parser.parse_args()

    set_seed(42)
    dist.init_process_group(backend='nccl')
    world_size=dist.get_world_size()    # 程序使用的总进程数
    global_rank=dist.get_rank()
    local_size=torch.cuda.device_count()    # 当前节点上存在几张显卡
    local_rank=int(os.environ["LOCAL_RANK"]) # 进程在当前节点上的序号
    torch.cuda.set_device(local_rank%local_size) # 确保进程在多个GPU上是平分的


    action_list=generate_action_list(world_size=world_size,num_stages=args.num_stages,num_chunks=args.num_chunks)[global_rank]
    print('rank = '+str(global_rank)+'action_list = '+str(action_list))


    # 输出是三个维度(batch_size,seq_length,vocab_size),经过一个softmax层，得到和y一样的维度(batch_size,seq_length)
    correct_result=torch.rand(args.batch_size*args.num_iterations,args.seq_length) # 用来计算LOSS
    
    module_list=generate_module(args) # 每个stage得拿多个module,因为有prefetchs
    model_list=copy.deepcopy(module_list)
    model=generate_model(args)
    pipeline=Pipeline(module_list,world_size,global_rank,local_rank,args.batch_size,args.num_chunks,args.seq_length,
    args.embedding_dim,args.ff_dim,args.num_stages)

    torch.cuda.synchronize()
    start_time=time.time()
  

    for i in range(args.num_iterations):
        pipeline.optimizer.zero_grad()
        pipeline.run_pipeline(action_list)
        dist.barrier()
        pipeline.optimizer.step()
        if global_rank==0:
            print(f"--------------- finish training step {i}")
 


    torch.cuda.synchronize()
    training_time=time.time()-start_time

    if global_rank==0:
        print("training time = {}".format(training_time))


    # if global_rank==0:
    #     another_optimizer=my_optimizer(model_list)
    #     start_time=time.time()
    #     # complete model
    #     another_device=torch.device(f'cuda:{0}')
    #     for it in range(args.num_iterations):
    #         another_optimizer.zero_grad()
    #         for i in range(args.num_chunks):
    #             pipeline.input_data[i]=pipeline.input_data[i].to(another_device)
    #             pipeline.input_data[i].requires_grad_()
    #             input_data=pipeline.input_data[i]
    #             for m in range(len(model_list)):
    #                 if m==len(model_list)-8:
    #                     last_data=input_data
    #                     last_data.retain_grad()
    #                 model_list[m]=model_list[m].to(another_device)
    #                 output=model_list[m](input_data)
    #                 input_data=output
    #                 input_data.requires_grad_()
    #             torch.autograd.backward(output.sum())
    #             print("continue grad",i,last_data.grad.shape,last_data.grad)
    #             # print("continue output ",output)
            
    #         another_optimizer.step()
            
               
    #     finish_time=time.time()
    #     training_time=finish_time-start_time
    #     print("baseline training time = {}".format(training_time))


    if global_rank==0:
        another_optimizer=my_optimizer(model_list)
        merged_model=merge_transformer_models(model_list,hidden_dim=args.embedding_dim,nhead=args.num_heads,ff_dim=args.ff_dim,dropout=0.0,norm_first=False,use_fp16=False)
        merged_model.to(torch.device(f'cuda:{0}'))
        start_time=time.time()
        for it in range(args.num_iterations):
            for i in range(args.num_chunks):
                pipeline.input_data[it][i].to(torch.device(f'cuda:{0}'))    
                pipeline.input_data[it][i].requires_grad_(True)
                output=merged_model(pipeline.input_data[it][i])
                torch.autograd.backward(output.mean())
                # print("continue model output",it,i,output)

            another_optimizer.step()
            # print("continue model grad",pipeline.input_data[-1].grad)
        finish_time=time.time()
        training_time=finish_time-start_time
        print("baseline training time = {}".format(training_time))

    
    
        
