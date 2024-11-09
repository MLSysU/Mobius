import torch
import torch.nn as nn
import time
from Runtime import *
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

set_seed(42)

# 实验代码
device_gpu = 'cuda:0'
device_cpu = 'cpu'

# 创建模型并将其移动到 GPU 上
model = LargeModel()
optimizer = optim.Adam(model.parameters(), lr=0.001)
model.to(device_gpu)
print_memory_status(device_gpu)

inputs = torch.randn(64, 1024, device=device_gpu)
outputs = model(inputs)

# print("After moving model to GPU:")
# print_memory_status(device_gpu)

# 将模型移动到 CPU
state_dict=offload1(model)
# model.to('cpu')
# outputs.to(device_cpu)

torch.cuda.empty_cache()
# print("\nAfter moving model to CPU:")
# print_memory_status(device_gpu)

#将模型移动到 GPU
reload1(model,state_dict)

targets = torch.randn(64, 2, device=device_gpu)

loss_fn = nn.MSELoss()
loss = loss_fn(outputs, targets)
optimizer.zero_grad() 
loss.backward()
print("\nAfter LOSS:")
print_memory_status(device_gpu)

for name, param in model.named_parameters():
    print("before step",param.grad)

state_dict=offload1(model)
print("\nAfter moving param:")
print_memory_status(device_gpu)

# for name, param in model.named_parameters():
#     print("after step",state_dict[name].grad)

torch.cuda.synchronize()
optimizer.step()

# 输出的梯度和param.grad是一样的
# for name, param in model.named_parameters():
#     print("after step",state_dict[name].grad)

reload1(model,state_dict)
# 获取 L1 层的梯度


