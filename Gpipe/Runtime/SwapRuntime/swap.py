import torch
from typing import Any
import time
import copy
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
        # torch.cuda.empty_cache()
        return output

    @staticmethod
    def backward(ctx, grad_output):
        # 卸载模型
        input, = ctx.saved_tensors
        grad_input = grad_output * 2
        return grad_input

def offload(model,cpu_model,offload_stream):
    # model offload, clear the memory storage occupied by model params in GPU.
    with torch.cuda.stream(offload_stream):
        with torch.profiler.record_function("offload model"):
            with torch.no_grad():
                for name, param in model.named_parameters():
                    if param.is_cuda:
                        param.data.untyped_storage().resize_(0)
                        if param.grad is not None:
                            cpu_model_param=dict(cpu_model.named_parameters())[name]
                            # double_grad=param.grad.to(dtype=torch.float32,device='cpu').detach()
                            # #double_grad.div_(1024)
                            # cpu_model_param.grad=torch.empty_like(double_grad,device='cpu').copy_(double_grad)
                            cpu_model_param.grad = torch.empty_like(param.grad, device='cpu', dtype=torch.float32).copy_(param.grad)
                            param.grad.untyped_storage().resize_(0)
                            param.grad=None
                model=None
    offload_stream.synchronize()
    # torch.cuda.empty_cache()
    return 


def load(model,cpu_model):
    # model reload, resize the model params' memory storage and fill it with params copy in CPU which is also referred by optimizer.
    # still working...
    cpu_model_dict=dict(cpu_model.named_parameters())
    for name,param in model.named_parameters():
        if param.is_cuda:
            cpu_tensor=cpu_model_dict[name]
            half_tensor_data=cpu_tensor.data.to(dtype=torch.float16,device='cpu')
            param.data.untyped_storage().resize_(half_tensor_data.untyped_storage().size())
            param.data.copy_(half_tensor_data)
            if cpu_tensor.grad is not None:
                half_tensor_grad=cpu_tensor.grad.to(dtype=torch.float16,device='cpu')
                # param.grad.untyped_storage().resize_(half_tensor.grad.untyped_storage().size())
                # param.grad.copy_(half_tensor.grad)
                param.grad=torch.empty_like(half_tensor_grad,device='cuda').copy_(half_tensor_grad)
    # torch.cuda.empty_cache()
    return 

def swap(module,device,local_module_list,load_stream):
    with torch.cuda.stream(load_stream):
        with torch.profiler.record_function("prefetch model"):
            next_module=copy.deepcopy(module)
            next_module.to(device)
            next_module.half()
            local_module_list.append(next_module)
    load_stream.synchronize()
                
  

    