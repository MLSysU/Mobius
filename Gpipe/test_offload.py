

# 对于一个GPU上面的变量，如果后续还有对该变量对应的GPU内存的引用，那么.to('cpu')就不会在GPU上面释放，譬如说后面需要求梯度
import torch
import torch.nn as nn
import time

# 检查 GPU 显存的使用情况
def print_memory_status(device):
    print(f"Memory allocated on {device}: {torch.cuda.memory_allocated(device) / (1024 * 1024):.2f} MB")
    print(f"Memory cached on {device}: {torch.cuda.memory_reserved(device) / (1024 * 1024):.2f} MB")

m = nn.Linear(10, 10)
inputs = [
    torch.randn(10, 10, device=0)
]
device_gpu = 'cuda:0'
device_cpu = 'cpu'
for input in inputs:
    m.to(device_gpu)
    activation = m(input)
    # 等待几秒钟，确保显存分配完成
    time.sleep(5)
    print("After moving model to GPU:")
    print_memory_status(device_gpu)
    m.to('cpu')
    # 等待几秒钟，确保显存状态更新
    time.sleep(5)
    print("\nAfter moving model to CPU:")
    print_memory_status(device_gpu)
    loss = activation.sum()
    # m.to(0) <--- without this copy, it doesn't work
    loss.backward()    
    # m.to('cpu')
    m.zero_grad()











