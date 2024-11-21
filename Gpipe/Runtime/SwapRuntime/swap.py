import torch
from typing import Any
import time
class SwapFunction(torch.autograd.Function):
    '''
    Planning to write a hook class for offlaod/reload. Still Working...
    '''
    @staticmethod
    def forward(ctx,module,device,input):
        # load model 
        module=module.to(device)
        # compute
        output=module(input)
        # offload model
        state_dict = {}
        pin_memory=True
        with torch.no_grad():  # 禁用梯度计算，加速操作并减少显存占用
            for name, param in model.named_parameters():
                if param.is_cuda:
                    # 复制参数到 CPU 并保存状态
                    src_tensor=param.data
                    cpu_backup = torch.empty(
                        src_tensor.size(),
                        dtype=src_tensor.dtype,
                        layout=src_tensor.layout,
                        device="cpu",
                        pin_memory=pin_memory,
                    )
                    cpu_backup.copy_(src_tensor, non_blocking=pin_memory)
                    state = (src_tensor.device, cpu_backup)
                    state_dict[name] = state
                    # 用 CPU 上的数据替换原参数，并释放 GPU 上的显存!!!
                    param.data.untyped_storage().resize_(0)
        ctx.state_dict=state_dict
        torch.cuda.empty_cache()
        return output

    @staticmethod
    def backward(ctx, grad_output):
        # 卸载模型
        input, = ctx.saved_tensors
        grad_input = grad_output * 2
        return grad_input

def offload(model,cpu_model):
    # model offload, clear the memory storage occupied by model params in GPU.
    with torch.no_grad():
        for name, param in model.named_parameters():
            if param.is_cuda:
                param.data.untyped_storage().resize_(0)
                if param.grad is not None:
                    cpu_model_param=dict(cpu_model.named_parameters())[name]
                    cpu_model_param.grad=torch.empty_like(param.grad,device='cpu').copy_(param.grad)
                    # param.grad.untyped_storage().resize_(0)
                    param.grad=None
    torch.cuda.empty_cache()
    return 


def load(model,cpu_model):
    # model reload, resize the model params' memory storage and fill it with params copy in CPU which is also referred by optimizer.
    # still working...
    cpu_model_dict=dict(cpu_model.named_parameters())
    for name,param in model.named_parameters():
        if param.is_cuda:
            cpu_tensor=cpu_model_dict[name]
            param.data.untyped_storage().resize_(cpu_tensor.untyped_storage().size())
            param.data.copy_(cpu_tensor)
            if cpu_tensor.grad is not None:
                # param.grad.untyped_storage().resize_(cpu_tensor.grad.untyped_storage().size())
                # param.grad.copy_(cpu_tensor.grad)
                param.grad=torch.empty_like(cpu_tensor.grad,device='cuda').copy_(cpu_tensor.grad)
    torch.cuda.empty_cache()
    return 

def offload2(model,cpu_model):
    # model offload, clear the memory storage occupied by model params in GPU.
    with torch.no_grad():
        for name, param in model.named_parameters():
            if param.is_cuda:
                print(name,param.shape)
                param.data.untyped_storage().resize_(0)
                if param.grad is not None:
                    cpu_model_param=dict(cpu_model.named_parameters())[name]
                    cpu_model_param.grad=torch.empty_like(param.grad,device='cpu').copy_(param.grad)
                    param.grad.untyped_storage().resize_(0)
    torch.cuda.empty_cache()
    return 


def load2(model,cpu_model):
    # model reload, resize the model params' memory storage and fill it with params copy in CPU which is also referred by optimizer.
    # still working...
    cpu_model_dict=dict(cpu_model.named_parameters())
    for name,param in model.named_parameters():
        if param.is_cuda:
            cpu_tensor=cpu_model_dict[name]
            param.data.untyped_storage().resize_(cpu_tensor.untyped_storage().size())
            if cpu_tensor.grad is not None:
                param.grad.untyped_storage().resize_(cpu_tensor.grad.untyped_storage().size())
                param.grad.copy_(cpu_tensor.grad)
    torch.cuda.empty_cache()
    return

'''
offload/reload function written for test_offload.py
'''
def offload1(model,cpu_model):
    # print("offload")
    for name, param in model.named_parameters():
        if param.is_cuda:
            param.data.untyped_storage().resize_(0)
            if param.grad is not None:
                cpu_model_param=dict(cpu_model.named_parameters())[name]
                cpu_model_param.grad=torch.empty_like(param.grad,device='cpu').copy_(param.grad)
                param.grad.untyped_storage().resize_(0)
    torch.cuda.empty_cache()
    return

def reload1(model,cpu_model):
    # print("reload")
    cpu_model_dict=dict(cpu_model.named_parameters())
    for name,param in model.named_parameters():
        if param.is_cuda:
            cpu_tensor=cpu_model_dict[name]
            param.data.untyped_storage().resize_(cpu_tensor.untyped_storage().size())
            param.data.copy_(cpu_tensor)
            if cpu_tensor.grad is not None:
                param.grad.untyped_storage().resize_(cpu_tensor.grad.untyped_storage().size())
                param.grad.copy_(cpu_tensor.grad)
    return 