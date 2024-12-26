import torch
import torch.nn as nn
import torch.optim as optim

# 创建一个简单的模型
class SimpleModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.linear = nn.Linear(10, 1)

    def forward(self, x):
        return self.linear(x)

model = SimpleModel()  # 模型在默认的 CPU 上
optimizer = optim.Adam(model.parameters(), lr=0.01)  # 创建优化器

# 查看优化器的初始状态
print("Before optimization:")
print(optimizer.state_dict())  # 此时 state 为空，因为还未调用 optimizer.step()

# 模拟前向传播和反向传播
model.to('cuda:0')  # 将模型迁移到 GPU
input = torch.randn(3, 10).to('cuda:0')
target = torch.randn(3, 1).to('cuda:0')
criterion = nn.MSELoss()

# 前向传播
output = model(input)
loss = criterion(output, target)

# 反向传播并优化
loss.backward()
optimizer.step()  # 更新参数

# 查看更新后的优化器状态
print("\nAfter optimization:")
print(optimizer.state_dict())

# 检查优化器状态的位置
for param in optimizer.state:
    print(f"Parameter device: {param.device}")
    for state_key, state_value in optimizer.state[param].items():
        print(f"  State '{state_key}' device: {state_value.device}")
