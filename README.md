# Mobius  
## execution method 
```
$ cd Gpipe 
$ CUDA_VISIBLE_DEVICES=0,2,1,3 torchrun  --nproc_per_node 4 ./main.py --seq_length=128 --num_stages=8 --num_iterations=4
```
ps: When running this file on your own machine, please annotate line62-line70 && anti-annotate line52-line60 in main.py to import model from huggingface instead of local cache.
