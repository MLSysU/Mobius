# Mobius  
## execution method 
```
$ cd Gpipe 
$ CUDA_VISIBLE_DEVICES=0,1,2,3 torchrun  --nproc_per_node 4 --master_port 29502 ./main.py --num_iterations=20   --batch_size=64 --num_stage=8 --no_prefetch --no_offload   
```
ps: When running this file on your own machine, please annotate line62-line70 && anti-annotate line52-line60 in main.py to import model from huggingface instead of local cache.
