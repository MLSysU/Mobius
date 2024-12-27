import torch

# device group setting
dist.init_process_group(backend='nccl')
world_size=dist.get_world_size()    # 程序使用的总进程数
global_rank=dist.get_rank()
local_size=torch.cuda.device_count()    # 当前节点上存在几张显卡
local_rank=int(os.environ["LOCAL_RANK"]) # 进程在当前节点上的序号
torch.cuda.set_device(local_rank%local_size) # 确保进程在多个GPU上是平分的

data = torch.rand(16,512,4096).to("cuda") # seq_length, batch_size/num_chunks, hidden_size 512, 16, 4096




