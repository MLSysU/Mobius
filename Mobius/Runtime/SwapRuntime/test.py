# Python 侧使用接口
import parallel
import torch
import torch.nn as nn
from torch.jit import ScriptModule

class OffloadManager:
    def __init__(self):
        self.offloader = parallel.get_offloader()
        # 创建专用 CUDA 流
        self.offload_stream = torch.cuda.Stream()

    def offload(self, model, cpu_model):
        # 将模型转换为 ScriptModule
        script_model = torch.jit.script(model)
        script_cpu_model = torch.jit.script(cpu_model)
        trace_model = torch.jit.trace(model, torch.randn(1, 1024).cuda())
        trace_cpu_model = torch.jit.trace(cpu_model, torch.randn(1, 1024))
        
        # 异步提交到 C++ 线程
        with torch.cuda.stream(self.offload_stream):
            self.offloader.submit(trace_model, trace_cpu_model)
            
        # 主线程可继续执行其它操作
        return self.offload_stream

# 使用示例
if __name__ == "__main__":
    manager = OffloadManager()
    
    # GPU 版 Sequential 模型
    gpu_model = nn.Sequential(
        nn.Linear(1024, 1024)
    ).cuda()  # 将整个 Sequential 移动到 GPU

    # CPU 版 Sequential 模型
    cpu_model = nn.Sequential(
        nn.Linear(1024, 1024)
    ).cpu()  # 将整个 Sequential 移动到 CPU

    
    script_gpu_model = torch.jit.script(gpu_model)  # 或 torch.jit.trace
    script_cpu_model = torch.jit.script(cpu_model)  # 或 torch.jit.trace

    print(script_gpu_model._c)
    print(isinstance(script_gpu_model, torch.jit.ScriptModule))  # 结果为 True
    # 执行卸载
    stream = manager.offload(script_gpu_model, script_cpu_model)

    
    
    # 需要同步时
    stream.synchronize()
