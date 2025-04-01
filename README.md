# Mobius  
## execution method 
```
$ cd Gpipe 
$ CUDA_VISIBLE_DEVICES=0,2,1,3 torchrun  --nproc_per_node 4 ./main.py --use_prefetch --use_offload --seq_length=256 --num_stages=16 --num_iterations=10
```
ps: When running this file on your own machine, please annotate line62-line70 && anti-annotate line52-line60 in main.py to import model from huggingface instead of local cache.
