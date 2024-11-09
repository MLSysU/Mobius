# Mobius  
## execution method  
cd Gpipe  
CUDA_VISIBLE_DEVICES=0,1,2,3 torchrun  --nproc_per_node 4 ./main.py  
ps: When running this file on your own machine, please annotate line62-line70 && anti-annotate line52-line60 in main.py to import model from huggingface instead of local cache.
