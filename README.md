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

1. You can directly use the docker file or install your environment locally according to the environment.yml.
2. Change the settings in the fine_tune.yaml according to your practical needs. Here are some descriptions about the functions of all tunable parameters.
3. Run the train.sh.
`bash train.sh`
### 🔬 Fine-tuning LLaMA-2-7B on XSum Dataset

This example demonstrates how to fine-tune the LLaMA-2-7B model on the XSum dataset using our MMoC-Pipe system with the provided configuration file.



## 🤝 Contributing

We welcome contributions from the community. 

## 📧 Contacts
If you have any questions, please raise an issue or contact us at liuhm59@mail2.sysu.edu.cn






