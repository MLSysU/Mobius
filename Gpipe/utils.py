from transformer.model.transformer_lm import TransformerLM,SequentialWithCustomForward
from torch import nn
import torch


def verify_peak_memory(rank):
    '''
    Method to evaluate the effect of mobius offload-reload trick,
    namely to evaluate the peak memory during the whole fine-tuning process.
    '''
    print(
        "cuda {:d}: Peak memory {:.2f} GB ; Persistent memory {:.2f} GB"
            .format(rank,torch.cuda.memory_stats(rank)["allocated_bytes.all.peak"] / 2**30 , torch.cuda.memory_stats(rank)["allocated_bytes.all.current"] / 2**30)
    )


def print_memory_status(device):
    '''
    Evaluate the memory usage at the current time.
    '''
    print(f"Memory allocated on {device}: {torch.cuda.memory_allocated(device) / (1024 * 1024):.2f} MB")
    print(f"Memory reserved on {device}: {torch.cuda.memory_reserved(device) / (1024 * 1024):.2f} MB")


def generate_module(args,config,layers_list):
    '''
    Generate models for every stage according to the layers_list extraced from Llama-2-7b.
    The partition strategy is average dividing.
    input: layers_list
    output: module_list
    ''' 
    module_list=[]
    layer_number_per_stage=[0]*args.num_stages
    # average_layers=config.num_hidden_layers//args.num_stages
    average_layers=config.num_hidden_layers//args.num_stages

    # Distribute layers to each stage
    start_idx = 0
    for stage_id in range(args.num_stages):
        if stage_id == args.num_stages - 1:
            end_idx = config.num_hidden_layers  # Last stage takes the remainder of the layers
            # end_idx=8
        else:
            end_idx = start_idx + average_layers
        stage_layers = layers_list[start_idx:end_idx]
        stage_model = SequentialWithCustomForward(*stage_layers)
        module_list.append(stage_model)
        start_idx = end_idx
    return module_list


def generate_module1(args):
    '''
    Generate models for every stage using self-defined TransformerLM which is composed of self-defined TransformerDecoderLayer.
    Every stage has the same number of TransformerDecoderLayer, which simulates the average partitioning strategy.
    input:
    output: module_list:list
    '''
    # 自己定义的model
    module_list=[]
    layer_number_per_stage=[0]*args.num_stages
    # 这里先持保留态度，看看各个stage的模型要如何划分，暂时先写成平均分的
    average_layers=24//args.num_stages
    for stage_id in range(args.num_stages):
        if stage_id==args.num_stages-1:
            layer_number=24-average_layers*(args.num_stages-1)
        else:
            layer_number=average_layers
        layer_number_per_stage[stage_id]=layer_number
        module=TransformerLM(hidden_dim=args.embedding_dim,nhead=args.num_heads,ff_dim=args.ff_dim,dropout=0.0,ndecoder=layer_number,norm_first=False,use_fp16=False,stage_id=stage_id)
        module_list.append(module)        
    return module_list



import torch.nn as nn

def merge_transformer_models(model_list, hidden_dim, nhead, ff_dim, dropout, norm_first=False, use_fp16=False, stage_id=-1):
    '''
    Concatenate some self-defined TransformerLM into one model.
    Design this function to fine-tune a complete model in only one device, setting comparison experiment to pipeline/mobius method.
    '''
    # 创建一个空列表，用于存储所有 TransformerDecoderLayer 层
    merged_layers = []
    
    for model in model_list:
        for layer in model:
            merged_layers.append(layer)
    merged_model = nn.Sequential(*merged_layers)
    
    # 将新的层组合封装成一个完整的 TransformerLM 类
    class MergedTransformerLM(TransformerLM):
        def __init__(self):
            super().__init__(hidden_dim, nhead, ff_dim, dropout, len(merged_layers), norm_first, use_fp16, stage_id)
            self.layers = merged_model

        def forward(self, inputs, chunk_id=None):
            return self.layers(inputs)

    return MergedTransformerLM()



def my_optimizer(model_list):
    '''
    input: module_list
    output: Adam optimizer
    '''
    parameters=[]
    for module in model_list:
        module_parameter=list(module.parameters())  # parameters()是nn.Module自带的函数，返回一个生成器，可以迭代调用出模型每一层的参数大小
        parameters+=module_parameter
    return torch.optim.Adam(parameters,lr=0.0001,weight_decay=1e-3)







