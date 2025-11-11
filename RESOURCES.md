# Curated Learning Resources for NVFP4 Kernel Hackathon

This document contains all the best resources for learning GPU kernel optimization, organized by topic.

---

## 🎯 Competition-Specific Resources

### Official Competition
- **Event Page**: https://lu.ma/9n27uem4
- **Discord Server**: https://discord.gg/gpumode
  - Channel: #nvidia-competition
- **Popcorn CLI**: https://github.com/gpu-mode/popcorn-cli
- **Reference Kernels**: https://github.com/gpu-mode/reference-kernels

### NVFP4 Format
1. **Primary**: [Introducing NVFP4 for Efficient Inference](https://developer.nvidia.com/blog/introducing-nvfp4-for-efficient-and-accurate-low-precision-inference/)
   - Best explanation of NVFP4 format
   - Includes diagrams and examples
   - Read this 2-3 times!

2. **Advanced**: [NVFP4 Training Paper](https://developer.nvidia.com/blog/nvfp4-trains-with-precision-of-16-bit-and-speed-and-efficiency-of-4-bit/)
   - How NVFP4 is used for training
   - More technical depth

3. **Comparison**: [NVFP4 vs Other 4-bit Formats](https://medium.com/data-science-collective/nvfp4-same-accuracy-with-2-3x-higher-throughput-for-4-bit-llms-03518ecba108)
   - Benchmarks against AWQ, GPTQ
   - Real-world performance numbers

---

## 📚 CUDA Programming Fundamentals

### For Complete Beginners (Start Here!)

1. **NVIDIA's "An Even Easier Introduction to CUDA"**
   - Link: https://developer.nvidia.com/blog/even-easier-introduction-cuda/
   - Time: 20 minutes
   - What you'll learn: Basic kernel structure, thread indexing

2. **CUDA Programming Guide (Chapter 2)**
   - Link: https://docs.nvidia.com/cuda/cuda-c-programming-guide/index.html#programming-model
   - Time: 1 hour
   - What you'll learn: Programming model, memory hierarchy
   - **Read sections**: 2.1-2.5 first

3. **GPU MODE "Introduction to CUDA" Lecture**
   - Link: https://www.youtube.com/@GPUMODE (search for CUDA intro)
   - Time: 45 minutes
   - What you'll learn: Real-world perspective from experts

### Intermediate CUDA

1. **CUDA Best Practices Guide**
   - Link: https://docs.nvidia.com/cuda/cuda-c-best-practices-guide/
   - Focus on: Chapters 9 (Memory Optimizations) and 10 (Execution Configuration)
   - This is your optimization bible!

2. **"How to Access Global Memory Efficiently in CUDA C/C++"**
   - Link: https://developer.nvidia.com/blog/how-access-global-memory-efficiently-cuda-c-kernels/
   - Essential for understanding coalescing

3. **"Using Shared Memory in CUDA C/C++"**
   - Link: https://developer.nvidia.com/blog/using-shared-memory-cuda-cc/
   - Bank conflicts, optimal patterns

---

## 🎓 GEMV-Specific Tutorials

### Essential Reading (In Order)

1. **"Learning CUDA by Optimizing SGEMV"** ⭐ START HERE
   - Link: https://maharshi.bearblog.dev/optimizing-sgemv-cuda/
   - Best step-by-step GEMV optimization guide
   - Shows progression from naive to optimized
   - Code examples included

2. **"CUDA HGEMV Optimization"**
   - Link: https://bruce-lee-ly.medium.com/nvidia-cuda-core-cuda-hgemv-optimization-51c25927ad43
   - Half-precision GEMV (closer to FP4!)
   - Multiple optimization strategies
   - Performance comparisons

3. **"GPU Matrix-Vector Product"**
   - Link: https://www.bealto.com/gpu-gemv.html
   - OpenCL but concepts transfer
   - Clear explanations of different approaches

### Code Examples

1. **FastGEMV** (GitHub)
   - Link: https://github.com/wangsiping97/FastGEMV
   - High-performance GEMV kernels
   - Good for studying optimization techniques

2. **cuda_hgemv** (GitHub)
   - Link: https://github.com/Bruce-Lee-LY/cuda_hgemv
   - Half-precision GEMV
   - Multiple kernel variants to study

3. **Gemlite** (GitHub + Blog)
   - Link: https://github.com/mobiusml/gemlite
   - Blog: https://mobiusml.github.io/gemlite_blogpost/
   - Low-bit quantized GEMV
   - Most relevant to FP4!

---

## 🔧 Development Tools

### Profiling & Debugging

1. **NVIDIA Nsight Compute**
   - Link: https://developer.nvidia.com/nsight-compute
   - Essential for finding bottlenecks
   - Tutorial: https://www.youtube.com/watch?v=GVVqHjZnVXA

2. **CUDA-GDB Debugging**
   - Link: https://docs.nvidia.com/cuda/cuda-gdb/
   - For when kernels crash or produce wrong results

3. **Compute Sanitizer**
   - Link: https://docs.nvidia.com/cuda/compute-sanitizer/
   - Find memory errors, race conditions

### Libraries & Frameworks

1. **CUTLASS** (NVIDIA's Template Library)
   - Link: https://github.com/NVIDIA/cutlass
   - Examples: https://github.com/NVIDIA/cutlass/tree/main/examples
   - Check `examples/92_gemv` for reference!

2. **Triton** (Python-based kernels)
   - Link: https://github.com/openai/triton
   - Tutorial: https://triton-lang.org/main/getting-started/tutorials/
   - Easier than raw CUDA, still high performance

---

## 📖 Advanced Topics (Week 2+)

### Blackwell Architecture

1. **Blackwell Architecture Whitepaper**
   - Link: https://resources.nvidia.com/en-us-blackwell-architecture
   - Tensor Core details, memory hierarchy

2. **Compute Capabilities**
   - Link: https://docs.nvidia.com/cuda/cuda-c-programming-guide/index.html#compute-capabilities
   - Find sm_100 (Blackwell) specifications

### Tensor Core Programming

1. **"Programming Tensor Cores in CUDA 9"**
   - Link: https://developer.nvidia.com/blog/programming-tensor-cores-cuda-9/
   - Foundation for understanding tensor cores

2. **CUTLASS GEMM Explained**
   - Link: https://developer.nvidia.com/blog/cutlass-linear-algebra-cuda/
   - How to use CUTLASS for optimal GEMMs

### Low-Bit Quantization

1. **"Quantization Fundamentals"**
   - Link: https://huggingface.co/blog/hf-bitsandbytes-integration
   - Understand quantization concepts

2. **"Block-wise Quantization"**
   - Link: Search for "GPTQ paper" or "AWQ paper"
   - Similar to NVFP4's block structure

---

## 🎬 Video Lectures (GPU MODE & Others)

### Must-Watch Lectures

1. **GPU MODE Lecture Series**
   - Channel: https://www.youtube.com/@GPUMODE
   - Playlist: Start with "CUDA programming basics"
   - Watch in order through "Memory optimization"

2. **NVIDIA GTC Talks**
   - Search: "NVIDIA GTC 2024 Blackwell"
   - Look for talks on:
     - "Optimizing for Blackwell"
     - "NVFP4 Deep Dive"
     - "Tensor Core Programming"

3. **"How to Optimize a CUDA Matrix Multiplication Kernel"**
   - Various on YouTube
   - Concepts apply to GEMV too!

---

## 💻 Example Repositories to Study

### Production-Quality Kernels

1. **cuBLAS** (closed source, but documentation)
   - Docs: https://docs.nvidia.com/cuda/cublas/
   - Study API and performance tips

2. **PyTorch CUDA Kernels**
   - Link: https://github.com/pytorch/pytorch/tree/main/aten/src/ATen/native/cuda
   - Real-world examples
   - Search for "gemv" or "gemm"

3. **FlashAttention**
   - Link: https://github.com/Dao-AILab/flash-attention
   - Not GEMV but amazing optimization techniques
   - Memory efficiency patterns

### Learning-Focused Repos

1. **CUDA Matrix Multiplication Optimization**
   - Link: https://github.com/leimao/CUDA-GEMM-Optimization
   - Step-by-step optimization
   - Blog: https://leimao.github.io/article/CUDA-Matrix-Multiplication-Optimization/

2. **Programming Massively Parallel Processors Examples**
   - Book examples online
   - Fundamentals + optimizations

---

## 📊 Understanding Performance

### Roofline Model

1. **"Understanding Performance with Roofline Model"**
   - Link: https://docs.nvidia.com/deeplearning/performance/dl-performance-matrix-multiplication/index.html
   - Understand memory vs compute bounds
   - CRITICAL for GEMV!

2. **GEMV is Memory-Bound!**
   - Arithmetic intensity ~1 FLOP/byte
   - Focus on bandwidth, not compute

### Profiling Metrics

**Key metrics to track:**
1. **Memory Bandwidth Utilization** (most important!)
   - Target: >80% of theoretical peak
   - Check with Nsight Compute

2. **Occupancy**
   - Target: >50%
   - Balance threads vs registers/shared memory

3. **Coalescing Efficiency**
   - Target: >90%
   - Measure global memory access pattern

---

## 🤝 Community & Getting Help

### Discord Channels
- **#nvidia-competition**: Competition-specific questions
- **#cuda-help**: General CUDA questions
- **#kernel-optimization**: Optimization strategies

### How to Ask Good Questions
1. **Share code snippet** (not full file)
2. **Describe what you tried** 
3. **Include error message** or performance numbers
4. **State your goal** (e.g., "trying to improve memory bandwidth")

### Learn from Others
- Watch what others are discussing
- Study approaches that work
- Don't be afraid to ask "why?"

---

## 📝 Documentation Quick Reference

### CUDA Documentation
- **Programming Guide**: https://docs.nvidia.com/cuda/cuda-c-programming-guide/
- **Best Practices**: https://docs.nvidia.com/cuda/cuda-c-best-practices-guide/
- **Nsight Compute**: https://docs.nvidia.com/nsight-compute/
- **cuBLAS**: https://docs.nvidia.com/cuda/cublas/

### NVIDIA Blogs
- **Main Blog**: https://developer.nvidia.com/blog
- Filter by: "CUDA", "Blackwell", "NVFP4"

---

## 🗓️ Suggested Reading Schedule

### Week 1 (Foundations)
- Day 1-2: CUDA intro resources, GEMV basics
- Day 3-4: Memory hierarchy, coalescing
- Day 5-6: SGEMV optimization blog, FastGEMV code
- Day 7: NVFP4 blog post

### Week 2 (Optimization)
- Day 8-9: Best Practices Guide Chapter 9
- Day 10-11: Shared memory blogs, bank conflicts
- Day 12-13: Gemlite blog, FP4 dequantization
- Day 14: Nsight Compute profiling guide

### Week 3 (Advanced)
- Day 15-16: CUTLASS examples, Tensor Cores
- Day 17: Blackwell architecture specifics
- Day 18: Last-minute optimizations

---

## 🎯 Key Takeaways for GEMV

### The Golden Rules
1. **GEMV is MEMORY-BOUND** - Optimize data movement, not compute
2. **Coalescing is CRITICAL** - Adjacent threads = adjacent memory
3. **Shared memory helps** - Reuse data, reduce global memory traffic
4. **Vectorize loads** - float4 > float for bandwidth
5. **Profile first** - Don't guess where the bottleneck is

### Common Mistakes to Avoid
1. ❌ Optimizing compute when memory is the bottleneck
2. ❌ Not profiling before optimizing
3. ❌ Copying optimizations without understanding
4. ❌ Giving up when first attempt is slow
5. ❌ Not asking for help in Discord

---

## 🚀 Bonus Resources

### Fun & Motivation
- **GPU MODE Podcast**: Interviews with experts
- **NVIDIA Blog Posts**: Success stories
- **Hackathon Write-ups**: Others' experiences

### After the Competition
- **Next competitions**: Keep building skills
- **Open source contributions**: CUTLASS, PyTorch
- **Research papers**: Latest optimization techniques

---

## 📧 Keep This Handy!

Bookmark this file! You'll reference it throughout the competition.

When stuck, check:
1. Is it a CUDA basics issue? → Section: CUDA Fundamentals
2. Is it a GEMV-specific issue? → Section: GEMV Tutorials  
3. Is it an NVFP4 issue? → Section: NVFP4 Format
4. Is it a performance issue? → Section: Understanding Performance

**Remember**: Every expert was once a beginner. You've got this! 💪

---

Last Updated: November 10, 2025
Competition: NVIDIA Developer Kernel Hackathon - Kernel #1
