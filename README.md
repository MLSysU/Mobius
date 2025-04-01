# Mobius  
## execution method 
```
$ This is an implementation of ZeRO-Offload.
$ cd /data/home/liuhuimin/mobius/Gpipe 
$ conda activate mobius
$ CUDA_VISIBLE_DEVICES=0,1,2,3 torchrun  --nproc_per_node 4 --master_port 29502 ./main.py --num_iterations=10 --batch_size=64 --num_stage=4 --no_prefetch --no_offload 
$ tensorboard --logdir=./test_log --load_fast=false  
```
ps: When running this file on your own machine, please annotate line62-line70 && anti-annotate line52-line60 in main.py to import model from huggingface instead of local cache.
