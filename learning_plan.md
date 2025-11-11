# NVFP4 Batched GEMV Learning Plan

**Duration**: 18 days (Nov 10 - Nov 28, 2025)
**Level**: Entry-level to competitive
**Goal**: Submit a competitive NVFP4 Batched GEMV kernel

## Learning Philosophy

As an entry-level developer with no kernel experience, you'll follow a **progressive learning approach**:
1. **Understand first, optimize later**
2. **Start simple, add complexity gradually**
3. **Measure everything - no guessing**
4. **Learn from the community**

## Week 1: Foundations (Nov 10-16)

### Day 1-2: Setup & Core Concepts
**Goal**: Environment ready, understand what you're building

#### Activities
- [x] Install Popcorn CLI
- [ ] Register for the hackathon
- [ ] Join GPU MODE Discord (#nvidia-competition channel)
- [ ] Set up GitHub repo
- [ ] Read: What is GEMV?
- [ ] Read: CUDA programming model basics
- [ ] Watch: GPU MODE intro videos

#### Key Concepts to Master
1. **Matrix-Vector Multiplication**
   - What: `y = A * x` where A is MxK matrix, x is K-length vector
   - Why: Core operation in neural networks (especially during inference)
   - Batched: Do multiple GEMV operations independently

2. **GPU Architecture Basics**
   - Streaming Multiprocessors (SMs)
   - Warps (32 threads execute together)
   - Memory hierarchy: Global → Shared → Registers

3. **NVFP4 Format**
   - E2M1 structure: 1 sign + 2 exponent + 1 mantissa bits
   - Values: Approximately -6 to +6
   - Two-level scaling:
     * FP8 scale per 16 values (micro-block)
     * FP32 scale per tensor

#### Resources
- CUDA C Programming Guide: Chapters 1-3
- NVFP4 Blog Post: Read 2-3 times until it clicks
- GPU MODE YouTube: "Introduction to CUDA" lecture

#### Deliverable
- Complete environment setup
- Write a 1-page summary of GEMV operation
- Understand NVFP4 memory layout

---

### Day 3-4: CUDA Programming Basics
**Goal**: Write your first CUDA kernel (not GEMV yet!)

#### Activities
- [ ] Install CUDA toolkit (for local testing/compilation)
- [ ] Write "Hello World" kernel
- [ ] Write vector addition kernel
- [ ] Understand thread indexing (threadIdx, blockIdx, blockDim)
- [ ] Practice calculating global thread ID

#### Key Concepts
1. **Kernel Launch Configuration**
   ```cuda
   kernel<<<numBlocks, threadsPerBlock>>>(args);
   ```

2. **Thread Indexing**
   ```cuda
   int tid = blockIdx.x * blockDim.x + threadIdx.x;
   ```

3. **Memory Management**
   - cudaMalloc / cudaFree
   - cudaMemcpy
   - Host vs Device memory

#### Hands-On Projects
1. **Vector Addition**
   - Input: Two vectors A[N], B[N]
   - Output: C[N] = A[N] + B[N]
   - Each thread computes one element

2. **Vector Dot Product** (this is closer to GEMV!)
   - Input: Two vectors A[N], B[N]
   - Output: scalar = sum(A[i] * B[i])
   - Use atomic operations or reduction

#### Resources
- CUDA C Programming Guide: Chapter 3 (Programming Interface)
- Practice: CUDA by Example book (online)

#### Deliverable
- Working vector addition kernel
- Working dot product kernel
- Understand thread-block-grid hierarchy

---

### Day 5-6: Memory Hierarchy Deep Dive
**Goal**: Understand why memory access patterns matter

#### Key Concepts
1. **Memory Types**
   - **Global Memory**: 
     * Largest, slowest (~400-600 cycles)
     * All threads can access
   - **Shared Memory**:
     * Per-block, fast (~20-30 cycles)
     * Explicitly managed by programmer
   - **Registers**:
     * Per-thread, fastest (1 cycle)
     * Compiler manages

2. **Coalesced Memory Access**
   - Adjacent threads access adjacent memory locations
   - Crucial for bandwidth utilization
   - Example:
     ```cuda
     // GOOD: Coalesced
     data[threadIdx.x] = ...;
     
     // BAD: Strided
     data[threadIdx.x * stride] = ...;
     ```

3. **Memory Bandwidth**
   - GEMV is **memory-bound** (not compute-bound)
   - Bottleneck: Moving data from memory
   - Goal: Maximize memory bandwidth utilization

#### Activities
- [ ] Read about memory coalescing
- [ ] Implement matrix transpose (good memory pattern practice)
- [ ] Use NVIDIA Visual Profiler to see memory access patterns
- [ ] Measure memory bandwidth

#### Exercises
1. **Matrix Transpose**
   - Naive version (uncoalesced)
   - Optimized version (coalesced)
   - Compare performance

2. **Shared Memory Practice**
   - Rewrite dot product using shared memory
   - Implement parallel reduction

#### Resources
- CUDA Best Practices Guide: Section on Memory Optimizations
- Blog: "How to Access Global Memory Efficiently in CUDA C/C++"

#### Deliverable
- Understand why GEMV is memory-bound
- Can explain coalescing in your own words
- Basic profiling skills

---

### Day 7: Naive GEMV Implementation
**Goal**: Get something working, even if slow

#### The Naive Approach
For batched GEMV: `Y[batch][m] = A[batch][m][k] * X[batch][k]`

**Strategy**: One thread per output element
```cuda
__global__ void naive_gemv_kernel(
    float* Y,      // Output [batch_size][M]
    float* A,      // Matrix [batch_size][M][K]
    float* X,      // Vector [batch_size][K]
    int M, int K, int batch_size
) {
    int batch = blockIdx.z;
    int m = blockIdx.x * blockDim.x + threadIdx.x;
    
    if (batch < batch_size && m < M) {
        float sum = 0.0f;
        for (int k = 0; k < K; k++) {
            sum += A[batch * M * K + m * K + k] * X[batch * K + k];
        }
        Y[batch * M + m] = sum;
    }
}
```

#### Activities
- [ ] Implement naive batched GEMV (FP32 first, not FP4 yet)
- [ ] Test with small matrices (verify correctness)
- [ ] Measure baseline performance
- [ ] Submit to leaderboard (even if slow!)

#### Deliverable
- Working naive GEMV kernel
- First leaderboard submission
- Baseline performance numbers

---

## Week 2: Optimization (Nov 17-23)

### Day 8-9: Memory Coalescing & Vectorization
**Goal**: Make memory accesses efficient

#### Optimization 1: Coalesced Access
The naive kernel has **terrible memory access patterns**:
- Each thread reads an entire row of A (non-coalesced)
- Solution: Transpose A or change access pattern

#### Optimization 2: Vectorized Memory Access
Instead of reading 1 float at a time, read 4 floats:
```cuda
float4 data = *((float4*)&A[offset]);
// Now data.x, data.y, data.z, data.w contain 4 values
```

#### Activities
- [ ] Analyze memory access pattern with profiler
- [ ] Implement vectorized loads (float2, float4)
- [ ] Measure improvement
- [ ] Compare with naive version

#### Expected Improvement
- 1.5-2x speedup from vectorization alone

---

### Day 10-11: Shared Memory Optimization
**Goal**: Reduce global memory traffic

#### Strategy: Cache the Input Vector
The input vector X is reused for every output element:
```cuda
__shared__ float X_shared[K];

// Load X into shared memory cooperatively
for (int i = threadIdx.x; i < K; i += blockDim.x) {
    X_shared[i] = X[batch * K + i];
}
__syncthreads();

// Now compute using X_shared instead of X
```

#### Activities
- [ ] Implement shared memory version
- [ ] Handle shared memory bank conflicts
- [ ] Measure improvement
- [ ] Profile shared memory usage

#### Expected Improvement
- 2-3x speedup over naive (cumulative with vectorization)

---

### Day 12-13: NVFP4 Format Implementation
**Goal**: Implement the actual 4-bit format

#### Understanding NVFP4 Layout
Each 16-value block stores:
- 16 × 4-bit values (8 bytes)
- 1 × FP8 scale (1 byte)  → Total: 9 bytes per block
- Plus: 1 × FP32 scale per tensor (4 bytes total)

#### Dequantization Formula
```
actual_value = fp4_value * fp8_scale * fp32_scale
```

#### Activities
- [ ] Study reference implementation of FP4 dequantization
- [ ] Implement FP4 unpacking
- [ ] Apply two-level scaling
- [ ] Verify correctness with reference

#### Key Challenges
1. **Bit Packing**: 2 values per byte
2. **Scale Application**: When to apply which scale?
3. **Precision**: Accumulate in FP32, not FP4!

---

### Day 14: Benchmark & Profile
**Goal**: Measure everything, identify bottlenecks

#### Activities
- [ ] Submit all versions to leaderboard
- [ ] Compare: naive vs optimized vs reference
- [ ] Use NVIDIA Nsight for profiling
- [ ] Identify top bottleneck

#### Metrics to Track
- **Latency**: Time per GEMV operation
- **Throughput**: Operations per second
- **Bandwidth**: GB/s achieved vs theoretical
- **Occupancy**: % of GPU utilized

---

## Week 3: Polish & Submit (Nov 24-28)

### Day 15-16: Advanced Optimizations
**Goal**: Squeeze out every last drop of performance

#### Possible Optimizations
1. **Warp-Level Primitives**
   - Use `__shfl_down_sync()` for reductions
   - Warp-level parallelism

2. **Thread-Block Tuning**
   - Experiment with different block sizes
   - 128 threads? 256? 512?

3. **Multiple Elements Per Thread**
   - Each thread computes 2-4 output elements
   - Better instruction-level parallelism

4. **Tensor Core Utilization** (Advanced!)
   - Blackwell has native FP4 tensor cores
   - Requires CUTLASS or inline PTX

#### Activities
- [ ] Implement at least 2 advanced optimizations
- [ ] Measure impact of each
- [ ] Combine best techniques

---

### Day 17: Final Testing & Validation
**Goal**: Ensure correctness, no bugs

#### Testing Checklist
- [ ] Correctness: Compare with CPU reference
- [ ] Edge cases: Small/large batch sizes
- [ ] Boundary conditions: Non-multiple-of-16 sizes
- [ ] Numerical stability: Check for NaN/Inf

---

### Day 18: Final Submission (Nov 28)
**Goal**: Submit your best work!

#### Pre-Submission Checklist
- [ ] Code is clean and commented
- [ ] Performance is validated
- [ ] Submitted to leaderboard
- [ ] Backup submission saved

---

## Daily Routine (Throughout)

### Morning (2 hours)
- Read documentation / watch tutorials
- Learn new concepts

### Afternoon (3-4 hours)
- Hands-on coding
- Implement new features

### Evening (1-2 hours)
- Profile and benchmark
- Document learnings

---

## Key Resources (Quick Reference)

### Must-Read Documentation
1. [CUDA C Programming Guide](https://docs.nvidia.com/cuda/cuda-c-programming-guide/)
2. [CUDA Best Practices Guide](https://docs.nvidia.com/cuda/cuda-c-best-practices-guide/)
3. [NVFP4 Blog Post](https://developer.nvidia.com/blog/introducing-nvfp4-for-efficient-and-accurate-low-precision-inference/)

### Video Lectures
1. [GPU MODE YouTube](https://www.youtube.com/@GPUMODE)
2. NVIDIA GTC Talks on Blackwell

### Example Implementations
1. [FastGEMV](https://github.com/wangsiping97/FastGEMV)
2. [cuda_hgemv](https://github.com/Bruce-Lee-LY/cuda_hgemv)
3. [Gemlite](https://github.com/mobiusml/gemlite)

### Community Support
- **GPU MODE Discord**: #nvidia-competition channel
- Ask questions early and often!
- Learn from others' approaches

---

## Success Metrics

### Week 1 Success
- ✓ Environment works
- ✓ Naive kernel runs
- ✓ Understand CUDA basics

### Week 2 Success  
- ✓ 3-5x faster than naive
- ✓ NVFP4 format working
- ✓ In top 50% of leaderboard

### Week 3 Success
- ✓ Top 25% of leaderboard (stretch: top 10%)
- ✓ Learned tons about GPU programming
- ✓ Ready for Kernel #2!

---

## Important Mindset Tips

1. **Don't compare to experts** - You're learning, they've done this for years
2. **Every bug is a learning opportunity** - Debug with curiosity
3. **Profile before optimizing** - Don't guess where the bottleneck is
4. **Start simple** - A working slow kernel beats a broken fast one
5. **Ask for help** - The community is friendly!

---

## Emergency Resources

If stuck on:
- **CUDA basics**: CUDA by Example book
- **Memory patterns**: CUDA Best Practices Guide Section 9-10
- **NVFP4 format**: Re-read the blog post, check reference implementation
- **Debugging**: Use printf in kernel (slow but helpful!)
- **Performance**: Profile with Nsight, find the bottleneck

---

**Remember**: The goal is to learn GPU programming, not just win. Even if you don't place top 3, you'll gain skills that will serve you for years. Have fun! 🚀
