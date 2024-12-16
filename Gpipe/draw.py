import matplotlib.pyplot as plt
import numpy as np
iterations=[10,20,50]
num_stages_list=[8,16]
batch_size_list=[64,128]
time_stage_eight_64=[327.7622730731964,654.8691167831421,1508.7693083286285] # batch size 64
time_stage_eight_128=[349.3065357208252,701.8608026504517,1626.7258546352386] # batch size 128
time_stage_sixteen_128=[345.4099340438843,1253.4993090629578,3138.805278301239]   # batch size 128
memory_stage_eight_64=[15.35,15.35,15.35]
memory_stage_eight_128=[26.87,26.87,26.87]
memory_stage_sixteen_128=[25.86,25.86,25.86]

# 绘图
plt.figure(figsize=(10, 6))

# 绘制折线图
plt.plot(iterations, memory_stage_eight_64, marker='o', label='Stage Eight Batch Size 64')
plt.plot(iterations, memory_stage_eight_128, marker='s', label='Stage Eight Batch Size 128')
plt.plot(iterations, memory_stage_sixteen_128, marker='^', label='Stage Sixteen Batch Size 128')

# 在图中标注数据点的值
for i, memory in enumerate(memory_stage_eight_64):
    plt.text(iterations[i], memory, f"{memory:.2f}", fontsize=10, ha='right')
for i, memory in enumerate(memory_stage_eight_128):
    plt.text(iterations[i], memory, f"{memory:.2f}", fontsize=10, ha='right')
for i, memory in enumerate(memory_stage_sixteen_128):
    plt.text(iterations[i], memory, f"{memory:.2f}", fontsize=10, ha='right')

# 设置图例
plt.legend(fontsize=12)

# 设置标题和坐标轴标签
plt.title('Iteration vs memory', fontsize=16)
plt.xlabel('Iteration', fontsize=14)
plt.ylabel('Tmemory (GB)', fontsize=14)

# 显示网格
plt.grid(alpha=0.5)

# 保存图像
plt.tight_layout()
plt.savefig('memory.png')