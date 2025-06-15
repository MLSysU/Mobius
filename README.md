# MMoC-Pipe: Memory-Efficient Full Fine-Tuning for Large Language Models

## 🚀 Introduction

This repository contains the official implementation of **MMoC-Pipe** (Memory-efficient Multi-stage offloading and Communication-optimized Pipeline), a novel system for full fine-tuning of large language models in resource-constrained environments. Our approach is detailed in the paper *"基于流水线和卸载的低资源大模型全量精调优化"* (Memory-Efficient Full Fine-Tuning via Pipeline and Offloading for Large Models under Resource Constraints).

## 🎯 Problem Statement

Full fine-tuning of large language models (LLMs) faces significant challenges in low-resource environments:
- **Memory bottleneck**: Models like LLaMA-2 (7B/13B) require substantial GPU memory
- **Limited hardware**: Commercial GPUs with restricted memory and bandwidth
- **Training efficiency**: Need for optimal resource utilization and faster convergence

## 💡 Our Solution

MMoC-Pipe addresses these challenges through an innovative combination of **pipeline parallelism** and **dynamic memory offloading**:

### Key Innovations
- **Multi-stage Memory Management**: Intelligent offloading between CPU-GPU heterogeneous storage
- **Communication-Computation Overlap**: Optimized pipeline execution with reduced idle time
- **PCIe-aware Stage Mapping**: Strategic placement of pipeline stages to minimize communication overhead
- **Multi-threaded Execution**: Concurrent data movement and computation for improved throughput

## 📊 Performance Highlights

Our experimental results demonstrate significant improvements over existing methods:

- **60% GPU Memory Reduction** compared to GPipe(only using pipeline architecture)
- **25% Training Speed Improvement** over ZeRO-Offload
- **Successful Full Fine-tuning** of LLaMA-2 (7B/13B) on 4×L20 commercial GPUs
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
- High-bandwidth CPU-GPU interconnect (PCIe recommended)

## 📚 Getting Started

1. You can directly use the docker file or install your environment locally according to the environment.yml.
2. You can change the settings in the fine_tune.yaml according to your practical needs. Here are some descriptions about the functions of all tunable parameters.
3. Run the train.sh.
`bash train.sh`
### 🔬 Fine-tuning LLaMA-2-7B on XSum Dataset

This example demonstrates how to fine-tune the LLaMA-2-7B model on the XSum dataset using our MMoC-Pipe system with the provided configuration file.



## 🤝 Contributing

We welcome contributions from the community. Please see our [contribution guidelines](CONTRIBUTING.md) for more details.

## 📄 License
## 📧 Contacts







