# NVFP4 Batched GEMV Optimization Guide

## Problem Overview

**Task**: Optimize batched matrix-vector multiplication for NVIDIA B200 GPU
- Input: `a` (M×K×L) in NVFP4, `b` (1×K×L) in NVFP4
- Output: `c` (M×1×L) in FP16
- Scale factors for block-scaled quantization

## Understanding NVFP4

NVFP4 (4-bit floating point, e2m1 format):
- 1 sign bit, 2 exponent bits, 1 mantissa bit
- Extremely compact but needs block-based scaling factors (FP8) for accuracy
- 16 elements share one FP8 scale factor (sf_vec_size = 16)

## Performance Targets (Speed of Light)

| M | K | L | Target Time (μs) |
|------|-------|---|--------------|
| 7168 | 16384 | 1 | 8.622 |
| 4096 | 7168 | 8 | 17.275 |
| 7168 | 2048 | 4 | 4.317 |

## Optimization Strategy

### Step 1: Write a Basic CUDA Kernel (Week 1)

Start with a simple CUDA kernel that:
1. Uses PyTorch's C++ extension or CuTe library
2. Implements basic GEMV: each thread computes one output element
3. Properly handles the NVFP4 format and scaling

**Key considerations**:
- NVFP4 data is packed (2 elements per byte as float4_e2m1fn_x2)
- Scale factors need to be applied per 16-element block
- Need to use B200's specialized NVFP4 instructions

### Step 2: Memory Optimization

**Critical optimizations**:
1. **Coalesced memory access**: Ensure threads access consecutive memory
2. **Shared memory**: Load matrix tiles to shared memory
3. **Register blocking**: Keep frequently accessed data in registers
4. **Vectorized loads**: Use float4/int4 loads when possible

### Step 3: B200-Specific Optimizations

NVIDIA B200 (Blackwell) has:
- Enhanced tensor cores for FP4 operations
- Higher memory bandwidth
- Better FP8/FP4 accumulation

**Use CuTe (CUTLASS Cute Templates)**:
- Abstracts low-level tensor core programming
- Optimal for B200's architecture
- Built-in support for NVFP4

### Step 4: Batching Optimization

Since L (batch dimension) varies:
- L=1: Optimize single GEMV
- L=4,8: Optimize parallel processing across batches
- Consider grid-stride loops for flexible batch handling

## Implementation Approaches

### Approach A: PyTorch C++ Extension (Easier)
```python
# submission.py
import torch
import my_cuda_kernel  # Your compiled CUDA extension

def custom_kernel(data):
    a, b, sfa, sfb, sfa_permuted, sfb_permuted, c = data
    # Call your CUDA kernel
    my_cuda_kernel.nvfp4_gemv(a, b, sfa_permuted, sfb_permuted, c)
    return c
```

### Approach B: CuTe/CUTLASS (Advanced, Better Performance)
```cpp
// Use CUTLASS CuTe templates
// Leverage B200 tensor cores directly
```

### Approach C: Triton (Python-based, Medium Difficulty)
```python
import triton
import triton.language as tl

@triton.jit
def nvfp4_gemv_kernel(...):
    # Write kernel in Triton
```

## Recommended Starting Point

**For entry-level with no GPU experience**:

1. **Start with Triton** (Python-based, easier learning curve)
2. Study existing Triton GEMV kernels
3. Adapt for NVFP4 format
4. Gradually optimize

**Resources**:
- [Triton Documentation](https://triton-lang.org/)
- [CUTLASS Examples](https://github.com/NVIDIA/cutlass)
- GPU MODE Discord for help
- [PyTorch C++ Extensions Guide](https://pytorch.org/tutorials/advanced/cpp_extension.html)

## Development Workflow

```bash
# Local development (MacBook)
# 1. Write kernel code
# 2. Test syntax/logic locally (without GPU)

# Remote testing (via Popcorn CLI)
# 1. Submit to remote B200 hardware
# 2. Run benchmarks
# 3. Iterate based on results
```

## Debugging Strategy

1. Start simple: Get correctness first
2. Profile with `eval.py` in profile mode
3. Check memory access patterns
4. Optimize hot spots identified in profiling

## Week-by-Week Plan

### Week 1 (Nov 11-17): Learning & Basic Implementation
- Study NVFP4 format and CUDA basics
- Write simplest possible CUDA kernel that passes tests
- Submit first version (even if slow)

### Week 2 (Nov 18-24): Optimization
- Profile your kernel
- Optimize memory access patterns
- Add shared memory tiling
- Optimize for different batch sizes

### Week 3 (Nov 25-28): Final Push
- Fine-tune for B200 architecture
- Experiment with tensor cores
- Submit multiple versions and compare

## Next Steps

1. ✅ Set up development environment
2. ⬜ Choose implementation approach (Triton recommended for beginners)
3. ⬜ Write minimal working kernel
4. ⬜ Test on remote GPU via Popcorn CLI
5. ⬜ Iterate and optimize

Good luck! 🚀
