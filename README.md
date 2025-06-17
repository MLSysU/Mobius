# MMoC-Pipe: LLM Full Fine-tuning with Low Resources using Pipeline Parallelism, Offload Strategy and Cross-Mapping

## 🚀 Introduction

This repository contains the official implementation of **MMoC-Pipe** (LLM Full Fine-tuning with Low Resources using Pipeline Parallelism, Offload Strategy and Cross-Mapping), a novel system for full fine-tuning of large language models in resource-constrained environments (one node with low GPU memory and PCIe communication). 

## 🎯 Problem Statement

Full fine-tuning of large language models (LLMs) faces significant challenges in low-resource environments:
- **Memory bottleneck**: Even models like LLaMA-2 (7B/13B) require substantial GPU memory
- **Limited hardware**: Commercial GPUs with restricted memory and bandwidth
- **Training efficiency**: Need for optimal resource utilization and faster training speed

## 💡 Our Solution

MMoC-Pipe addresses these challenges through an innovative combination of **pipeline parallelism**, **dynamic memory offloading** and **cross-mapping**:

### Key Innovations
- **Multi-stage Per Device**: Traditional pipeline parallelism partitions model layers into N stages, with N indicating the number of GPUs. We instead adopt virtual pipeline strategy, which partitions model layers into v*N stages, with every GPU executing v pipeline stages in turns. Take N=4, v=2 for example. The 8 virtual stages (v×N=2×4) are distributed cyclically among the 4 GPUs: GPU1 handles stages 1 and 5, GPU2 handles stages 2 and 6, GPU3 handles stages 3 and 7, and GPU4 handles stages 4 and 8.
- **Model Offload**: We dynamically offload model parameters into CPU during the training process, saving HBM for activations and gradients.
- **Communication-Computation Overlap**: Optimized pipeline execution with reduced idle time 
- **Cross Mapping**: Strategic placement of pipeline stages to minimize communication overhead
- **Multi-stream && Multi-threaded Execution**: Concurrent data movement and computation for improved throughput

### System Architecture
![MMoC-Pipe Architecture](./assets/architecture.png)

## 📊 Performance Highlights

Our experimental results demonstrate significant improvements over existing methods:

- **Successful Full Fine-tuning** of LLaMA-2 (7B/13B) on 4×L20 commercial GPUs
- **60% GPU Memory Reduction** compared to GPipe(only using pipeline architecture)
  ![Comparison on memory occupation to baselines](./assets/memory-occup.png)
- **25% Training Speed Improvement** over ZeRO-Offload
  ![Comparison on training speed to baselines](./assets/training-time.png)
- **Enhanced Scalability** for larger batch sizes and sequence lengths


## 🎯 Target Use Cases

This implementation is designed for:
- **Researchers** working with limited GPU resources
- **Developers** seeking efficient LLM fine-tuning solutions
- **Organizations** with budget constraints on high-end hardware
- **Academic institutions** with shared computing resources

## 🔧 System Requirements

- NVIDIA GPUs (tested on L20, adaptable to other models)
- CUDA-compatible environment
- Sufficient CPU memory for offloading operations
- CPU-GPU interconnect (PCIe supported)

## 📚 Getting Started

1. You can directly use the docker file below or install your environment locally according to the environment.yml.
   `docker pull coir1hat1man/mobius:latest`
3. Change the settings in the fine_tune.yaml according to your practical needs. Here are some descriptions about the functions of all tunable parameters.
    | Parameter | Value | Description |
    |-----------|-------|-------------|
    | `batch_size` | 64 | Number of samples processed in each training batch |
    | `num_chunks` | 4 | Number of micro-batch |
    | `seq_length` | 128 | Maximum sequence length for input tokens |
    | `embedding_dim` | 4096 | Dimension of the embedding layer |
    | `ff_dim` | 4096 | Dimension of the feed-forward network in transformer blocks |
    | `num_iterations` | 2 | Total number of training iterations to run |
    | `num_stages` | 8 | Number of pipeline stages for model parallelism |
    | `num_layers` | 32 | This parameter should be configured according to your model's specifications. You can also set it to a value less than the total number of layers in your model to simulate a smaller architecture.|
    | `num_heads` | 32 | Number of attention heads in multi-head attention |
    | `model` | "llama-2-7b-hf" | Model name on huggind-face |
    | `dataset` | "xsum" | Dataset name for fine-tuning, datasets must have been downloaded locally.|
    | `save_results` | "test_result.txt" | File path to save training information(including training time, memory usage) |
    | `save_dir` | "save_ckpt" | Directory to save model checkpoints |
    | `use_prefetch` | true | Enable data prefetching to improve training efficiency |
    | `use_offload` | true | Enable model parameter offloading to reduce GPU memory usage |
    | `cuda_visible_devices` | "0,2,1,3" | GPU device IDs to use for training (specific order for pipeline stages) |
    | `master_port` | 29502 | Port number for distributed training communication |
4. Run the train.sh.
`bash train.sh`
### 🔬 Example: Fine-tuning LLaMA-2-7B on XSum Dataset

This example demonstrates how to fine-tune the LLaMA-2-7B model on the XSum dataset using our MMoC-Pipe system with the provided configuration file.

1. use settings in the fine_tune.yaml.
2. run `bash train.sh`
3. get the ckpt
   
   <img src="./assets/ckpt.png" width="25%" alt="Checkpoint image">

## 🤝 Contributing

We welcome contributions from the community. 

## 📧 Contacts
If you have any questions, please raise an issue or contact us at liuhm59@mail2.sysu.edu.cn






