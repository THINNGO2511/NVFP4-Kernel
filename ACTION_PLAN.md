# Immediate Action Plan - NVFP4 GEMV Hackathon

## TODAY (Nov 11): Setup & First Submission ✅

### Step 1: Set up Popcorn CLI (30 minutes)

```bash
# On your MacBook
git clone https://github.com/gpu-mode/popcorn-cli
cd popcorn-cli

# Follow the installation instructions
# You'll need to authenticate and link your Discord account
```

### Step 2: Clone and Set Up Your Repo (15 minutes)

```bash
cd ~
git clone https://github.com/THINNGO2511/NVFP4-Kernel
cd NVFP4-Kernel

# Create directory structure
mkdir -p kernels/gemv
cd kernels/gemv

# Copy the reference files here
# You'll get these from: https://github.com/gpu-mode/reference-kernels/tree/main/problems/nvidia/nvfp4_gemv
```

### Step 3: Make Your First Submission (15 minutes)

**Goal**: Submit the reference implementation (even though it's slow) just to verify your setup works!

```bash
# Copy reference.py to submission.py
# Or use the PyTorch optimized version I created for you

# Test locally (syntax check)
python submission.py

# Submit via Popcorn CLI
popcorn submit nvidia/nvfp4_gemv submission.py

# Check results on the leaderboard
```

**Expected outcome**: Your submission should PASS correctness tests (but be slow).

---

## THIS WEEK (Nov 11-17): Learning & Basic Optimization 📚

### Day 1-2: Learn the Basics

**Study Materials** (prioritize these):

1. **Understanding NVFP4** (2 hours)
   - Read: https://developer.nvidia.com/blog/introducing-nvfp4-for-efficient-and-accurate-low-precision-inference/
   - Key concepts: 4-bit floating point, block scaling, why it's faster

2. **GEMV Basics** (1 hour)
   - Understand matrix-vector multiplication
   - Memory access patterns
   - Why it's memory-bound vs compute-bound

3. **Choose Your Path** (Pick ONE):

   **Option A: Triton (Recommended for beginners)** ⭐
   - Tutorial: https://triton-lang.org/main/getting-started/tutorials/02-fused-softmax.html
   - Study GEMV examples in Triton
   - Pros: Python-based, easier to learn
   - Cons: Might be slightly slower than raw CUDA

   **Option B: PyTorch C++ Extensions**
   - Tutorial: https://pytorch.org/tutorials/advanced/cpp_extension.html
   - Pros: Good balance of ease and performance
   - Cons: Need to learn C++/CUDA

   **Option C: Pure CUDA/CuTe**
   - CUTLASS repo: https://github.com/NVIDIA/cutlass
   - Pros: Maximum performance potential
   - Cons: Steep learning curve

### Day 3-4: Implement Basic Kernel

**Milestone**: Create a kernel that:
- ✅ Passes all correctness tests
- ✅ Handles NVFP4 format properly
- ✅ Applies scaling factors correctly
- ⏱️ Is at least 2x faster than reference

**Approach**:
1. Start with Triton template (I provided one)
2. Handle NVFP4 unpacking correctly
3. Apply block scaling
4. Test on remote GPU

### Day 5-7: Profile and Optimize

**Tools**:
```bash
# Profile your kernel
popcorn profile nvidia/nvfp4_gemv submission.py

# This will show you:
# - Memory bandwidth utilization
# - Compute utilization  
# - Bottlenecks
```

**Common optimizations**:
- Memory coalescing
- Shared memory usage
- Reduce CPU-GPU synchronization
- Vectorized loads/stores

---

## WEEK 2 (Nov 18-24): Advanced Optimization 🚀

### Focus Areas:

1. **B200-Specific Features**
   - Use tensor cores for NVFP4
   - Leverage hardware FP4 instructions
   - Optimize for Blackwell memory hierarchy

2. **Batch Optimization**
   - L=1: Optimize for single GEMV
   - L=4,8: Parallel batch processing
   - Dynamic dispatch based on batch size

3. **Kernel Tuning**
   - Experiment with tile sizes (BLOCK_M, BLOCK_K)
   - Thread block configuration
   - Register usage optimization

---

## WEEK 3 (Nov 25-28): Final Push 🏁

### Strategies:

1. **Multiple Submissions**
   - Submit different versions
   - Compare performance on benchmarks
   - Keep best version

2. **Learn from Discord**
   - Join discussions in #nvidia-competition channel
   - See what techniques others mention
   - Ask questions!

3. **Ensemble Approach**
   - Have different kernels for different problem sizes
   - Dispatch dynamically based on M, K, L

---

## Resources & Support 🛠️

### Essential Links:
- **Problem repo**: https://github.com/gpu-mode/reference-kernels/tree/main/problems/nvidia/nvfp4_gemv
- **Popcorn CLI**: https://github.com/gpu-mode/popcorn-cli
- **Your repo**: https://github.com/THINNGO2511/NVFP4-Kernel
- **Discord**: GPU MODE #nvidia-competition channel

### When You Get Stuck:
1. Check Discord #nvidia-competition channel
2. Look at GPU MODE YouTube lectures
3. Study CUTLASS examples for similar problems
4. Ask specific questions with error messages

### Performance Expectations:

**Week 1**: 2-5x faster than reference
**Week 2**: 5-10x faster than reference  
**Week 3**: 10-20x faster, approaching speed of light

**Competitive performance**: Within 2x of speed of light targets

---

## Immediate TODO List ☑️

- [ ] Set up Popcorn CLI and authenticate
- [ ] Clone reference kernels repo
- [ ] Make first submission (reference implementation)
- [ ] Verify submission appears on leaderboard
- [ ] Read NVFP4 blog post
- [ ] Choose learning path (Triton/CUDA/PyTorch)
- [ ] Join GPU MODE Discord and introduce yourself
- [ ] Create weekly schedule and stick to it

---

## Tips for Success 💡

1. **Submit early, submit often**: Don't wait for perfection
2. **Start simple**: Get correctness first, speed second
3. **Profile religiously**: Measure before optimizing
4. **Learn from failures**: Every slow submission teaches you something
5. **Community**: Use Discord! Others are learning too
6. **Time management**: You have 18 days - pace yourself

---

## Questions to Answer This Week

- [ ] How does NVFP4 packing work? (2 elements per byte)
- [ ] How are scale factors applied in block-scaled quantization?
- [ ] What's the memory access pattern for GEMV?
- [ ] Which dimension (M, K, or L) should I parallelize over?
- [ ] What's limiting performance: memory bandwidth or compute?

---

Good luck! Remember: This is a learning experience. Even if you don't win, you'll learn a ton about GPU programming! 🚀

**Next action**: Set up Popcorn CLI and make your first submission TODAY.
