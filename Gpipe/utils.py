from model.transformer_lm import TransformerLM
from torch import nn

def generate_module(args):
    module_list=[]
    layer_number_per_stage=[0]*args.num_stages
    # 这里先持保留态度，看看各个stage的模型要如何划分，暂时先写成平均分的
    average_layers=args.num_layers//args.num_stages
    for stage_id in range(args.num_stages):
        if stage_id==args.num_stages-1:
            layer_number=args.num_layers-average_layers*(args.num_stages-1)
        else:
            layer_number=average_layers
        layer_number_per_stage[stage_id]=layer_number
        module=TransformerLM(hidden_dim=args.embedding_dim,nhead=args.num_heads,ff_dim=args.ff_dim,dropout=0.0,ndecoder=layer_number,norm_first=False,use_fp16=False,stage_id=stage_id)

        module_list.append(module)        


    return module_list

def generate_model(args):
    model=TransformerLM(hidden_dim=args.embedding_dim,nhead=args.num_heads,ff_dim=args.ff_dim,dropout=0.0,ndecoder=args.num_layers,norm_first=False,use_fp16=False)
    return model

import torch.nn as nn

def merge_transformer_models(model_list, hidden_dim, nhead, ff_dim, dropout, norm_first=False, use_fp16=False, stage_id=-1):
    # 创建一个空列表，用于存储所有 TransformerDecoderLayer 层
    merged_layers = []
    
    # 遍历每个 TransformerLM 模型，提取其 TransformerDecoderLayer 层
    for model in model_list:
        # 逐层添加到 merged_layers 中
        for layer in model:
            # 每层添加时，创建唯一的 layer_id
            merged_layers.append(layer)
    
    # 使用所有提取的层创建新的 TransformerLM 模型
    merged_model = nn.Sequential(*merged_layers)
    
    # 将新的层组合封装成一个完整的 TransformerLM 类
    class MergedTransformerLM(TransformerLM):
        def __init__(self):
            super().__init__(hidden_dim, nhead, ff_dim, dropout, len(merged_layers), norm_first, use_fp16, stage_id)
            self.layers = merged_model

        def forward(self, inputs, chunk_id=None):
            return self.layers(inputs)

    return MergedTransformerLM()



