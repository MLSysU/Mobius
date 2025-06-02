import torch
import time

def measure_pcie_bandwidth(size_in_mb, num_trials=10):
    # 检查是否有至少一个 GPU
    if torch.cuda.device_count() < 1:
        print("At least one GPU is required to measure PCIe bandwidth.")
        return

    # 定义设备
    cpu_device = torch.device("cpu")
    gpu_device = torch.device("cuda:0")

    # 创建要传输的数据
    data_size = size_in_mb * 1024 * 1024 // 4  # 以 float32 为单位
    tensor_cpu = torch.ones(data_size, dtype=torch.float32, device=cpu_device)
    tensor_gpu = torch.zeros(data_size, dtype=torch.float32, device=gpu_device)

    # 测试 GPU -> CPU 传输带宽
    total_time_gpu_to_cpu = 0
    for _ in range(num_trials):
        # 确保 GPU 和 CPU 状态同步
        torch.cuda.synchronize()
        start = time.time()

        # 从 GPU 传输到 CPU
        tensor_gpu.to(cpu_device)

        # 同步并记录时间
        torch.cuda.synchronize()
        total_time_gpu_to_cpu += time.time() - start

    total_time_gpu_to_cpu = 0
    for _ in range(num_trials):
        # 确保 GPU 和 CPU 状态同步
        torch.cuda.synchronize()
        start = time.time()

        # 从 GPU 传输到 CPU
        tensor_gpu.to(cpu_device)

        # 同步并记录时间
        torch.cuda.synchronize()
        total_time_gpu_to_cpu += time.time() - start

    # 测试 CPU -> GPU 传输带宽
    total_time_cpu_to_gpu = 0
    for _ in range(num_trials):
        torch.cuda.synchronize()
        start = time.time()

        # 从 CPU 传输到 GPU
        tensor_cpu.to(gpu_device)

        torch.cuda.synchronize()
        total_time_cpu_to_gpu += time.time() - start

    total_time_cpu_to_gpu = 0
    for _ in range(num_trials):
        torch.cuda.synchronize()
        start = time.time()

        # 从 CPU 传输到 GPU
        tensor_cpu.to(gpu_device)

        torch.cuda.synchronize()
        total_time_cpu_to_gpu += time.time() - start

    # 计算平均时间和带宽
    avg_time_gpu_to_cpu = total_time_gpu_to_cpu / num_trials
    avg_time_cpu_to_gpu = total_time_cpu_to_gpu / num_trials
    bandwidth_gpu_to_cpu = size_in_mb / avg_time_gpu_to_cpu  # MB/s
    bandwidth_cpu_to_gpu = size_in_mb / avg_time_cpu_to_gpu  # MB/s

    print(f"Average PCIe Bandwidth (GPU -> CPU): {bandwidth_gpu_to_cpu:.2f} MB/s")
    print(f"Average PCIe Bandwidth (CPU -> GPU): {bandwidth_cpu_to_gpu:.2f} MB/s")

if __name__ == "__main__":
    # 设置数据大小和测试次数
    size_in_mb = 1024  # 数据大小 (MB)
    num_trials = 10    # 测试次数

    measure_pcie_bandwidth(size_in_mb, num_trials)
