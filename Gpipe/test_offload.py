import torch
import torch.nn as nn
import time
from Runtime import *
from torch.cuda import nvtx
import torch.optim as optim

def set_seed(seed):
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

# 检查 GPU 显存的使用情况
def print_memory_status(device):
    print(f"Memory allocated on {device}: {torch.cuda.memory_allocated(device) / (1024 * 1024):.2f} MB")
    print(f"Memory reserved on {device}: {torch.cuda.memory_reserved(device) / (1024 * 1024):.2f} MB")

# 创建一个简单的模型
class LargeModel(nn.Module):
    def __init__(self):
        super(LargeModel, self).__init__()
        
        # 增加层数和隐藏单元数量
        self.fc1 = nn.Linear(1024, 2048)  # L1层
        self.fc2 = nn.Linear(2048, 2048)  # L2层
        self.fc3 = nn.Linear(2048, 4096)  # L3层
        self.fc4 = nn.Linear(4096, 2048)  # L4层
        self.fc5 = nn.Linear(2048, 1024)  # L5层
        self.fc6 = nn.Linear(1024, 512)   # L6层
        self.fc7 = nn.Linear(512, 256)    # L7层
        self.fc8 = nn.Linear(256, 128)    # L8层
        self.fc9 = nn.Linear(128, 64)     # L9层
        self.fc10 = nn.Linear(64, 32)     # L10层
        self.fc11 = nn.Linear(32, 16)     # L11层
        self.fc12 = nn.Linear(16, 8)      # L12层
        self.fc13 = nn.Linear(8, 4)       # L13层
        self.fc14 = nn.Linear(4, 2)       # L14层
    
    def forward(self, x):
        # 逐层传递并使用 ReLU 激活函数
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        x = torch.relu(self.fc3(x))
        x = torch.relu(self.fc4(x))
        x = torch.relu(self.fc5(x))
        x = torch.relu(self.fc6(x))
        x = torch.relu(self.fc7(x))
        x = torch.relu(self.fc8(x))
        x = torch.relu(self.fc9(x))
        x = torch.relu(self.fc10(x))
        x = torch.relu(self.fc11(x))
        x = torch.relu(self.fc12(x))
        x = torch.relu(self.fc13(x))
        x = self.fc14(x)  # 输出层
        return x

def print_memory_status(device):
    '''
    Evaluate the memory usage at the current time.
    '''
    print(f"Memory allocated on {device}: {torch.cuda.memory_allocated(device) / (1024 * 1024):.2f} MB")
    print(f"Memory reserved on {device}: {torch.cuda.memory_reserved(device) / (1024 * 1024):.2f} MB")

set_seed(42)
nvtx.range_push("Matrix Multiplication")

# 实验代码
device_gpu = 'cuda:3'
device_cpu = 'cpu'

compute_stream=torch.cuda.Stream()
load_stream=torch.cuda.Stream()

start_load = torch.cuda.Event(enable_timing=True)
end_load = torch.cuda.Event(enable_timing=True)
start_compute = torch.cuda.Event(enable_timing=True)
end_compute = torch.cuda.Event(enable_timing=True)
a = torch.randn(4000, 4000, device="cuda:3").detach()
b = torch.randn(4000, 4000, device="cuda:3").detach()
x = torch.randn(4000, 4000, device="cuda:3").detach()
y = torch.randn(4000, 4000, device="cuda:3").detach()

torch.cuda.synchronize()
start_time=time.time()

for i in range(5):  # 分成两部分
    c = torch.matmul(a, b)
    a = c
print_memory_status(device_gpu)

for i in range(5):  # 分成两部分
    d = torch.matmul(x, y)
    x = d
print_memory_status(device_gpu)

# with torch.cuda.stream(load_stream):
#     start_load.record()
#     print(f"load_stream: {load_stream}")
#     for i in range(5):  # 分成两部分
#         c = torch.matmul(a, b)
#         a = c
#     end_load.record()
#     print_memory_status(device_gpu)
    

# with torch.cuda.stream(compute_stream):
#     start_compute.record()
#     print(f"compute_stream: {compute_stream}")
#     for i in range(5):  # 分成两部分
#         d = torch.matmul(x, y)
#         x = d
#     end_compute.record()
#     print_memory_status(device_gpu)

torch.cuda.synchronize()
# end_time=time.time()
# total_time = end_time - start_time  # 总时间（秒）
# load_time = start_load.elapsed_time(end_load)  # 加载模型的时间（毫秒）
# compute_time = start_compute.elapsed_time(end_compute)  # 前向传播的时间（毫秒）

# print(f"Total time: {total_time*1000:.2f} ms")
# print(f"Model loading time: {load_time:.2f} ms")
# print(f"Compute time: {compute_time:.2f} ms")
# print(f"add time: {load_time+compute_time:.2f} ms")

print(a)
print(x)  
print("real total time",time.time()-start_time)



'''
inputs = torch.randn(64, 1024, device=device_gpu)

# CPU创建模型并创建优化器
model = LargeModel()
next_model=LargeModel()
optimizer = optim.Adam(model.parameters(), lr=0.001)


# 模型复制，加载到GPU
start_load = torch.cuda.Event(enable_timing=True)
end_load = torch.cuda.Event(enable_timing=True)
start_compute = torch.cuda.Event(enable_timing=True)
end_compute = torch.cuda.Event(enable_timing=True)

start_time=time.time()
# 在 load_stream 上加载模型
with torch.cuda.stream(load_stream):
    start_load.record()
    module = copy.deepcopy(model)
    module.to(device_gpu)
    next_module = copy.deepcopy(next_model)
    next_module.to(device_gpu)
    end_load.record()

# 在 compute_stream 上执行前向传播
with torch.cuda.stream(compute_stream):
    start_compute.record()
    outputs = module(inputs)
    end_compute.record()

# 同步所有事件
torch.cuda.synchronize()

# 计算时间
total_time = time.time() - start_time  # 总时间（秒）
load_time = start_load.elapsed_time(end_load)  # 加载模型的时间（毫秒）
compute_time = start_compute.elapsed_time(end_compute)  # 前向传播的时间（毫秒）

print(f"Total time: {total_time*1000:.2f} ms")
print(f"Model loading time: {load_time:.2f} ms")
print(f"Compute time: {compute_time:.2f} ms")
print(f"add time: {load_time+compute_time:.2f} ms")


print(outputs)


# 卸载模型
offload1(module,model)


torch.cuda.empty_cache()
print("\nAfter offload:")
print_memory_status(device_gpu)

# 重新加载模型
reload1(module,model)

print("\nAfter reload:")
print_memory_status(device_gpu)

#反向传播
targets = torch.randn(64, 2, device=device_gpu)
loss_fn = nn.MSELoss()
loss = loss_fn(outputs, targets)
optimizer.zero_grad() 
loss.backward()
print("\nAfter LOSS:")
print_memory_status(device_gpu)



offload1(module,model)
print("\nAfter offload:")
print_memory_status(device_gpu)

# for name, param in model.named_parameters():
#     print("after step",state_dict[name].grad)

torch.cuda.synchronize()
optimizer.step()


# 输出的梯度和param.grad是一样的
# for name, param in model.named_parameters():
#     print("after step",state_dict[name].grad)

reload1(module,model)
print("\nAfter reload:")
print_memory_status(device_gpu)

outputs=module(inputs)
'''


