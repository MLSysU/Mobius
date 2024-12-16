import torch
import torch.nn as nn
import time

# 定义一个简单的大型神经网络
class LargeNet(nn.Module):
    def __init__(self):
        super(LargeNet, self).__init__()
        self.layer1 = nn.Linear(4096, 4096)
        self.layer2 = nn.Linear(4096, 4096)
        self.layer3 = nn.Linear(4096, 4096)
        self.layer4 = nn.Linear(4096, 4096)
        
    def forward(self, x):
        x = torch.relu(self.layer1(x))
        x = torch.relu(self.layer2(x))
        x = torch.relu(self.layer3(x))
        x = torch.relu(self.layer4(x))
        return x

# 初始化模型
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = LargeNet().to(device)

# 定义 CPU 上的数据
batch_size = 1024  # 将数据分为多个批次
num_batches = 4
input_data_gpu = torch.randn(num_batches, batch_size, 4096, device='cuda')

# 创建两个流：一个用于数据传输，一个用于计算
transfer_stream = torch.cuda.Stream()
compute_stream = torch.cuda.Stream()

# 创建事件列表用于在批次之间同步
events = [torch.cuda.Event(enable_timing=True) for _ in range(num_batches)]

# 开始重叠操作
for epoch in range(5):
    start_time = time.time()

    for batch_idx in range(num_batches):
        # 在传输流中将当前批次的数据从 CPU 拷贝到 GPU
        with torch.cuda.stream(transfer_stream):
            input_data_cpu = input_data_gpu[batch_idx].to('cpu', non_blocking=True)
            # 在传输完成后，记录一个事件，用于同步
            events[batch_idx].record(transfer_stream)

        # 在计算流中计算前一个批次（确保传输完成后再计算）
        if batch_idx > 0:  # 确保至少有一个批次已传输完成
            with torch.cuda.stream(compute_stream):
                events[-2].synchronize()  # 等待前一个批次的传输完成
                output = model(input_data_gpu_prev)

        # 保存当前批次的数据作为下一轮计算的数据
        input_data_gpu_prev = input_data_gpu

    # 处理最后一个批次的计算
    with torch.cuda.stream(compute_stream):
        events[-1].synchronize()  # 确保最后一个批次的传输完成
        output = model(input_data_gpu_prev)

    # 同步两条流，确保所有操作完成
    torch.cuda.synchronize()

    end_time = time.time()
    print(f"Epoch [{epoch+1}/5], Time: {end_time - start_time:.4f} seconds")
