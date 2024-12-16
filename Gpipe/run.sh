import time
import subprocess

commands = [
    # "CUDA_VISIBLE_DEVICES=0,1,2,3 torchrun  --nproc_per_node 4 ./main.py --num_iterations=20 --batch_size=128 --num_stage=16",
    # "CUDA_VISIBLE_DEVICES=0,1,2,3 torchrun  --nproc_per_node 4 ./main.py --num_iterations=30 --batch_size=128 --num_stage=16",
    "CUDA_VISIBLE_DEVICES=0,1,2,3 torchrun  --nproc_per_node 4 ./main.py --num_iterations=50 --batch_size=128 --num_stage=16",
]

# 执行每条命令并设置间隔
i=0
for cmd in commands:
    i=i+1
    print(f"Running: {cmd}")
    subprocess.run(cmd, shell=True)  # 或 os.system(cmd)
    time.sleep(800*i)  # 设置 10 秒的间隔
