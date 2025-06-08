# 读取 YAML 并导出变量
eval $(python -c '
import yaml
with open("fine_tune.yaml") as f:
    config = yaml.safe_load(f)
    for k, v in config.items():
        print(f"{k}={v}")
')

# 构造布尔参数
PREFETCH_FLAG=""
if [ "$use_prefetch" = "True" ]; then
    PREFETCH_FLAG="--use_prefetch"
else
    PREFETCH_FLAG="--no_prefetch"
fi

OFFLOAD_FLAG=""
if [ "$use_offload" = "True" ]; then
    OFFLOAD_FLAG="--use_offload"
else
    OFFLOAD_FLAG="--no_offload"
fi

# 运行命令
CUDA_VISIBLE_DEVICES=$cuda_visible_devices torchrun --nproc_per_node=4 --master_port=$master_port ./main.py \
    --num_iterations=$num_iterations \
    --batch_size=$batch_size \
    --num_stages=$num_stages \
    --num_layers=$num_layers \
    $PREFETCH_FLAG \
    $OFFLOAD_FLAG \
    --seq_length=$seq_length \
    --model=$model \
    --embedding_dim=$embedding_dim
