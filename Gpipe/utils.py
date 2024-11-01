from transformer.model.transformer_lm import TransformerLM,SequentialWithCustomForward
from torch import nn
import torch

def generate_module(args,config,layers_list):
    module_list=[]
    layer_number_per_stage=[0]*args.num_stages
    # average_layers=config.num_hidden_layers//args.num_stages
    average_layers=24//args.num_stages
    # Distribute layers to each stage
    start_idx = 0
    for stage_id in range(args.num_stages):
        if stage_id == args.num_stages - 1:
            end_idx = 24  # Last stage takes the remainder of the layers
        else:
            end_idx = start_idx + average_layers
        stage_layers = layers_list[start_idx:end_idx]
        stage_model = SequentialWithCustomForward(*stage_layers)
        module_list.append(stage_model)
        start_idx = end_idx
    return module_list

def generate_module1(args):
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
    parameters=[]
    for module in model_list:
        module_parameter=list(module.parameters())  # parameters()是nn.Module自带的函数，返回一个生成器，可以迭代调用出模型每一层的参数大小
        parameters+=module_parameter
    return torch.optim.Adam(parameters,lr=0.0001,weight_decay=1e-3)







