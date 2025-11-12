# NVFP4 Technical Reference

## What is NVFP4?

NVFP4 (NVIDIA 4-bit Floating Point) is a ultra-low precision number format designed for efficient ML inference on Blackwell GPUs.

### Format: E2M1 (4 bits total)
```
┌─────┬───────────┬─────────┐
│  S  │    E E    │    M    │
│ 1b  │   2 bits  │  1 bit  │
└─────┴───────────┴─────────┘
```

- **S**: Sign bit (1 bit)
- **E**: Exponent (2 bits) - biased by 1
- **M**: Mantissa (1 bit)

### Representable Values

With only 4 bits, NVFP4 can only represent 16 distinct values:

**Positive values** (S=0):
- 0.0 (special case: exp=00, mant=0)
- 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0

**Negative values** (S=1):
- Same magnitudes as positive, but negative

**Special values**:
- NaN (Not a Number)

### Storage: float4_e2m1fn_x2

In PyTorch, NVFP4 is stored as `torch.float4_e2m1fn_x2`:
- **Two 4-bit values packed into one byte** (8 bits)
- This is why K dimension must be even

```python
# Example: array of 8 NVFP4 values
# Stored in 4 bytes (uint8 array)
uint8_data = [byte0, byte1, byte2, byte3]

# Each byte contains 2 NVFP4 values:
# byte0 = [nvfp4_value_0, nvfp4_value_1]
# byte1 = [nvfp4_value_2, nvfp4_value_3]
# etc.
```

---

## Block-Scaled Quantization

### The Problem

NVFP4 alone has terrible precision:
- Only 16 distinct values
- Maximum value is 6.0
- Large gaps between representable numbers

**Solution**: Block-based scaling with FP8 scale factors!

### How Block Scaling Works

1. **Divide data into blocks** of 16 elements
2. **Store one FP8 scale factor per block**
3. **Actual value = NVFP4 value × FP8 scale factor**

```
Block of 16 NVFP4 values:
[v0, v1, v2, ..., v15]  (each in range ~[-6, 6])

Scale factor (FP8):
scale = 2.5

Actual values:
[v0×2.5, v1×2.5, v2×2.5, ..., v15×2.5]
```

### Scale Factor Layout

The scale factors have a complex memory layout optimized for B200's tensor cores:

**Input format** (what you receive):
- `sfa`: [M, K//16, L] - one scale per 16 K elements
- `sfb`: [1, K//16, L] - one scale per 16 K elements

**Permuted format** (optimized for hardware):
- `sfa_permuted`: [32, 4, rest_m, 4, rest_k, L]
- `sfb_permuted`: [32, 4, rest_n, 4, rest_k, L]

The permuted format aligns with how B200's tensor cores consume data.

---

## The GEMV Operation with Block Scaling

### Mathematical Formulation

For standard GEMV without scaling:
```
c[m] = Σ(k=0 to K-1) a[m,k] × b[k]
```

With NVFP4 block scaling:
```
c[m] = Σ(block=0 to K//16-1) Σ(i=0 to 15) 
       (a_nvfp4[m, block×16+i] × scale_a[m, block]) ×
       (b_nvfp4[block×16+i] × scale_b[block])
```

Simplified:
```
c[m] = Σ(block=0 to K//16-1) scale_a[m,block] × scale_b[block] ×
       Σ(i=0 to 15) a_nvfp4[m, block×16+i] × b_nvfp4[block×16+i]
```

**Key insight**: 
- Inner sum (over 16 elements) can be done in NVFP4
- Multiply by scales afterward
- This is how `torch._scaled_mm` works internally

---

## Memory Layout Details

### Matrix A: [M, K, L]
```
For batch l=0:
Row 0: [a[0,0], a[0,1], ..., a[0,K-1]]
Row 1: [a[1,0], a[1,1], ..., a[1,K-1]]
...
Row M-1: [a[M-1,0], ..., a[M-1,K-1]]

Storage: K-major order (K dimension is contiguous)
```

### Vector B: [1, K, L]
```
For batch l=0:
[b[0], b[1], b[2], ..., b[K-1]]

But padded to [128, K, L] for tensor core alignment
```

### Scale Factors: [M, K//16, L]
```
For each row m, batch l:
[scale[m,0], scale[m,1], ..., scale[m,K//16-1]]

One scale factor for every 16 elements in K dimension
```

---

## Optimization Considerations

### Memory Access Pattern

**Good pattern** (coalesced):
```cuda
// Thread i accesses consecutive K elements
for (int k = 0; k < K; k++) {
    float a_val = a[thread_m, k, batch];  // Coalesced across threads
    float b_val = b[0, k, batch];
    acc += a_val * b_val;
}
```

**Bad pattern** (strided):
```cuda
// Thread i accesses strided M elements  
for (int m = 0; m < M; m++) {
    float a_val = a[m, thread_k, batch];  // Strided! Slow!
    ...
}
```

### Parallelization Strategy

**Option 1: Parallelize over M** (recommended)
- Each thread/block computes one output element c[m]
- Good for large M (e.g., 7168)
- Memory access to A is coalesced

**Option 2: Parallelize over K**
- Multiple threads reduce over K for same output
- Requires final reduction step
- Good for small M, large K

**Option 3: Parallelize over L** (batches)
- Independent batches processed in parallel
- Good when L is large (e.g., L=8)

---

## NVFP4 Tensor Core Usage

B200 has specialized tensor cores for NVFP4 operations:

### Tensor Core GEMM Pattern
```
Block tile: [BLOCK_M, BLOCK_N, BLOCK_K]
- BLOCK_M, BLOCK_N: Output tile size (e.g., 128x128)
- BLOCK_K: K dimension tile (e.g., 64)

Each warp computes a sub-tile using WMMA (Warp Matrix Multiply Accumulate)
```

### CuTe Abstraction
```cpp
// Pseudo-code for CuTe GEMV
using MMA = MMA_Atom<
    FP4_E2M1,      // A type
    FP4_E2M1,      // B type  
    FP16,          // C type
    SM_100         // Blackwell architecture
>;

auto a_tile = make_tile(a_ptr, ...);
auto b_tile = make_tile(b_ptr, ...);
auto c_tile = make_tile(c_ptr, ...);

gemm(mma, a_tile, b_tile, c_tile, scales_a, scales_b);
```

---

## Reference Implementation Walkthrough

Let's understand the reference implementation:

```python
def ref_kernel(data: input_t) -> output_t:
    a_ref, b_ref, sfa_ref_cpu, sfb_ref_cpu, _, _, c_ref = data
    _, _, l = c_ref.shape
    
    # Loop over each batch
    for l_idx in range(l):
        # Convert scale factors to blocked format for torch._scaled_mm
        scale_a = to_blocked(sfa_ref_cpu[:, :, l_idx])
        scale_b = to_blocked(sfb_ref_cpu[:, :, l_idx])
        
        # Call PyTorch's scaled matrix multiply
        # (m, k) @ (n, k).T -> (m, n)
        # Since n=1 (padded to 128), this is GEMV
        res = torch._scaled_mm(
            a_ref[:, :, l_idx],          # [M, K]
            b_ref[:, :, l_idx].transpose(0, 1),  # [K, 128]
            scale_a.cuda(),
            scale_b.cuda(),
            bias=None,
            out_dtype=torch.float16,
        )
        
        # Extract first column (n=0) as our result
        c_ref[:, 0, l_idx] = res[:, 0]
    
    return c_ref
```

**Why it's slow**:
1. Loop over L prevents parallelization
2. CPU-GPU synchronization for scales
3. `torch._scaled_mm` is general-purpose, not optimized for GEMV
4. Overhead of Python loop

---

## Speed of Light Analysis

The "speed of light" targets are based on:

```
Time = max(
    compute_time_FFMA,
    memory_time_DRAM
)
```

### For M=7168, K=16384, L=1:

**FLOPs**: 
- 2 × M × K × L = 2 × 7168 × 16384 × 1 = 235M FLOPs

**Data**: 
- A: 7168 × 16384 × 0.5 bytes (NVFP4) = 58.7 MB
- B: 1 × 16384 × 0.5 bytes = 8 KB
- Scales: 7168 × 1024 × 1 byte (FP8) = 7.3 MB
- Total: ~66 MB

**B200 specs** (at 1.5 GHz):
- Compute: ~600 TFLOPS (for FP4)
- Memory BW: ~8 TB/s

**Compute time**: 235M / 600T = 0.39 μs
**Memory time**: 66 MB / 8 TB/s = 8.25 μs

**Bottleneck**: Memory-bound! 
**Target**: 8.622 μs (close to memory limit)

This tells you: optimize memory access patterns first!

---

## Common Bugs to Avoid

1. **Incorrect NVFP4 unpacking**
   - Must handle 2 values per byte correctly
   
2. **Wrong scale factor application**
   - Scales are per-block (16 elements), not per-element
   
3. **Incorrect permuted scale layout**
   - Using `sfa` instead of `sfa_permuted` with tensor cores
   
4. **Memory alignment issues**
   - NVFP4 requires specific alignment for tensor cores
   
5. **Not handling L dimension correctly**
   - Each batch is independent - can parallelize

---

## Testing Your Implementation

```python
# Basic correctness test
from reference import generate_input, check_implementation
from submission import custom_kernel

# Generate test input
data = generate_input(m=128, k=256, l=1, seed=42)

# Run your kernel
output = custom_kernel(data)

# Check correctness
is_correct, message = check_implementation(data, output)
print(f"Correct: {is_correct}, Message: {message}")
```

---

## Further Reading

1. **NVFP4 Blog**: https://developer.nvidia.com/blog/introducing-nvfp4-for-efficient-and-accurate-low-precision-inference/
2. **CUTLASS Documentation**: https://github.com/NVIDIA/cutlass
3. **Tensor Core Programming**: https://docs.nvidia.com/cuda/parallel-thread-execution/index.html#warp-level-matrix-instructions
4. **FP8 Specification**: https://docs.nvidia.com/cuda/cuda-c-programming-guide/index.html#floating-point-formats

---

## Quick Reference

| Property | Value |
|----------|-------|
| NVFP4 bits | 4 (1 sign, 2 exp, 1 mantissa) |
| Values per byte | 2 |
| Block size | 16 elements |
| Scale factor format | FP8 E4M3 |
| K divisibility | 64 (for testing) |
| M divisibility | Varies (see task.yml) |
| Output format | FP16 |
| Target GPU | NVIDIA B200 (Blackwell) |

---

Good luck with your kernel optimization! 🚀
