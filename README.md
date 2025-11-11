# NVFP4 Batched GEMV Kernel Optimization

Submission for the NVIDIA Developer Kernel Hackathon - Kernel #1: NVFP4 Batched GEMV

**Competition Period**: Nov 10 - Nov 28, 2025
**Target Hardware**: NVIDIA Blackwell B200

## Overview

This repository contains my implementation of an optimized NVFP4 Batched GEMV (General Matrix-Vector Multiplication) kernel for the GPU MODE hackathon.

### What is GEMV?
- **Operation**: `y = A * x`
  - `A`: Matrix (M × K)
  - `x`: Input vector (K)
  - `y`: Output vector (M)
- **Batched**: Multiple independent GEMV operations in parallel

### What is NVFP4?
NVFP4 is NVIDIA's 4-bit floating point format for Blackwell GPUs:
- **Structure**: E2M1 (1 sign bit, 2 exponent bits, 1 mantissa bit)
- **Range**: Approximately -6 to +6
- **Scaling**: Two-level approach
  - FP8 (E4M3) scale per 16-value micro-block
  - FP32 scale per tensor
- **Memory**: 3.5x reduction vs FP16, 1.8x vs FP8

## Learning Resources

### CUDA Programming Basics
- [CUDA C Programming Guide](https://docs.nvidia.com/cuda/cuda-c-programming-guide/index.html)
- [GPU MODE YouTube](https://www.youtube.com/@GPUMODE) - Weekly lectures from ML experts
- [NVIDIA NVFP4 Blog Post](https://developer.nvidia.com/blog/introducing-nvfp4-for-efficient-and-accurate-low-precision-inference/)

### GEMV Optimization Resources
- [Optimizing SGEMV](https://maharshi.bearblog.dev/optimizing-sgemv-cuda/)
- [CUDA HGEMV Optimization](https://bruce-lee-ly.medium.com/nvidia-cuda-core-cuda-hgemv-optimization-51c25927ad43)
- [FastGEMV](https://github.com/wangsiping97/FastGEMV)
- [Gemlite](https://mobiusml.github.io/gemlite_blogpost/)

## Development Setup

### Prerequisites
- macOS (local development)
- Python 3.8+
- Access to GPU MODE Discord
- Remote GPU access for benchmarking

### Installation

1. **Clone this repo**:
```bash
git clone https://github.com/THINNGO2511/NVFP4-Kernel.git
cd NVFP4-Kernel
```

2. **Install Popcorn CLI**:
```bash
# Download for your platform from:
# https://github.com/gpu-mode/popcorn-cli/releases

# Make executable (Mac/Linux)
chmod +x popcorn-cli

# Move to PATH
sudo mv popcorn-cli /usr/local/bin/
```

3. **Configure Popcorn CLI**:
```bash
# Get API URL from GPU MODE Discord
# Type: /get-api-url in Discord
export POPCORN_API_URL="your-api-url"

# Register with Discord (recommended)
popcorn-cli register discord

# Or register with GitHub
popcorn-cli register github

# Verify registration
cat $HOME/.popcorn.yaml
```

## Project Structure

```
NVFP4-Kernel/
├── README.md                 # This file
├── docs/                     # Documentation and notes
│   ├── learning_plan.md     # Week-by-week learning plan
│   └── notes/               # Daily learning notes
├── kernels/                  # Kernel implementations
│   ├── naive/               # Baseline naive implementation
│   ├── optimized/           # Optimized versions
│   └── final/               # Final submission
├── tests/                    # Test scripts
├── benchmarks/              # Benchmark scripts
└── submission.py            # Final submission file
```

## Development Workflow

### Local Development
- Write and test kernel code on macOS
- Use CPU emulation for basic testing
- Validate logic and structure locally

### Remote Testing
- Submit to GPU MODE infrastructure via Popcorn CLI
- Benchmark on actual Blackwell B200 hardware
- Iterate based on performance results

### Submission
```bash
# Submit to leaderboard
popcorn-cli submit --gpu B200 --leaderboard nvfp4_gemv --mode leaderboard submission.py
```

## Learning Plan

### Week 1 (Nov 10-16): Foundations
- Day 1-2: Setup environment, understand GEMV
- Day 3-4: CUDA basics (threads, blocks, warps)
- Day 5-6: Memory hierarchy (global, shared, registers)
- Day 7: Implement naive GEMV kernel

### Week 2 (Nov 17-23): Optimization
- Day 8-9: Memory coalescing and vectorization
- Day 10-11: Shared memory optimization
- Day 12-13: NVFP4 format implementation
- Day 14: Benchmark and profile

### Week 3 (Nov 24-28): Polish & Submit
- Day 15-16: Advanced optimizations
- Day 17: Final testing and validation
- Day 18: Submit to leaderboard

## Key Optimization Strategies

### Memory Access Patterns
- **Coalesced Memory Access**: Ensure adjacent threads access adjacent memory
- **Vectorized Loads**: Use float4, int4 for wider memory transactions
- **Shared Memory**: Cache frequently accessed data

### Thread Organization
- **Warp-Level Optimization**: Leverage 32-thread warp execution
- **Thread Block Configuration**: Balance occupancy vs resources
- **Register Usage**: Minimize to increase occupancy

### NVFP4-Specific
- **Block Scaling**: Handle 16-value micro-blocks with FP8 scales
- **Dual Scaling**: Apply both FP8 and FP32 scales correctly
- **Precision Management**: Accumulate in higher precision

## Resources & References

### Problem Reference
- [GPU MODE Reference Kernels](https://github.com/gpu-mode/reference-kernels/tree/main/problems/nvidia/nvfp4_gemv)

### Community
- [GPU MODE Discord](https://discord.gg/gpumode) - #nvidia-competition channel
- [Popcorn CLI Docs](https://github.com/gpu-mode/popcorn-cli)

### Papers & Documentation
- [NVFP4 Technical Blog](https://developer.nvidia.com/blog/introducing-nvfp4-for-efficient-and-accurate-low-precision-inference/)
- [CUDA C Best Practices Guide](https://docs.nvidia.com/cuda/cuda-c-best-practices-guide/)
- [Blackwell Architecture Whitepaper](https://resources.nvidia.com/en-us-blackwell-architecture)

## Progress Tracking

- [ ] Environment setup complete
- [ ] Understand GEMV operation
- [ ] Naive kernel implementation
- [ ] Coalesced memory access
- [ ] Shared memory optimization
- [ ] NVFP4 format implementation
- [ ] Vectorized memory access
- [ ] Warp-level optimizations
- [ ] First benchmark submission
- [ ] Performance tuning
- [ ] Final submission

## Notes

### Current Status
- **Date Started**: Nov 10, 2025
- **Days Remaining**: 18
- **Current Phase**: Setup

### Performance Goals
- Target: Approach "speed of light" performance
- Baseline: Beat reference implementation
- Stretch: Top 3 on leaderboard

## Contact & Support

- **GitHub**: [@THINNGO2511](https://github.com/THINNGO2511)
- **Discord**: GPU MODE server - #nvidia-competition
- **Competition**: [Luma Event Page](https://luma.com/9n27uem4)

