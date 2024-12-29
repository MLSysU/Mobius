import time
import subprocess

commands = [
    # "CUDA_VISIBLE_DEVICES=0,1,2,3 torchrun  --nproc_per_node 4 --master_port 29502 ./main.py --num_iterations=20 --batch_size=64 --num_stage=8 --use_prefetch",
    "CUDA_VISIBLE_DEVICES=0,1,2,3 torchrun  --nproc_per_node 4 --master_port 29502 ./main.py --num_iterations=20 --batch_size=64 --num_stage=8 --no_prefetch",
    "CUDA_VISIBLE_DEVICES=0,1,2,3 torchrun  --nproc_per_node 4 --master_port 29502 ./main.py --num_iterations=20 --batch_size=64 --num_stage=16 --use_prefetch",
    "CUDA_VISIBLE_DEVICES=0,1,2,3 torchrun  --nproc_per_node 4 --master_port 29502 ./main.py --num_iterations=20 --batch_size=64 --num_stage=16 --no_prefetch",
    # "CUDA_VISIBLE_DEVICES=0,1,2,3 torchrun  --nproc_per_node 4 --master_port 29502 ./main.py --num_iterations=20 --batch_size=64 --num_stage=8 --use_prefetch",  
    # "CUDA_VISIBLE_DEVICES=0,1,2,3 torchrun  --nproc_per_node 4 ./main.py --num_iterations=30 --batch_size=128 --num_stage=16",
    # "CUDA_VISIBLE_DEVICES=0,1,2,3 torchrun  --nproc_per_node 4 ./main.py --num_iterations=50 --batch_size=128 --num_stage=16",
]

for i, cmd in enumerate(commands):
    print(f"Running command {i + 1}/{len(commands)}: {cmd}")
    
    # 阻塞执行命令，直到命令完成
    result = subprocess.run(cmd, shell=True)
    
    # 检查命令执行是否成功
    if result.returncode == 0:
        print(f"Command {i + 1} finished successfully.")
    else:
        print(f"Command {i + 1} failed with return code {result.returncode}.")
        break  # 如果需要终止脚本，可以在失败时退出