# NVFP4 Kernel Hackathon - Getting Started Package

Welcome to your NVFP4 Batched GEMV optimization journey! This package contains everything you need to get started with the NVIDIA Kernel Hackathon.

## 📁 What's in This Package?

### 1. **ACTION_PLAN.md** ⭐ START HERE
Your immediate roadmap for the next 3 weeks. This tells you exactly what to do today, this week, and each subsequent week.

**Read this first!** It contains:
- Today's setup tasks
- Week-by-week breakdown
- Resource links
- Expected performance milestones

### 2. **NVFP4_TECHNICAL_REFERENCE.md** 📚
Deep technical dive into:
- NVFP4 format (4-bit floating point)
- Block-scaled quantization
- Memory layouts
- Speed of light analysis
- Common bugs to avoid

**When to read**: After completing setup, before writing code.

### 3. **NVFP4_GEMV_STARTER_GUIDE.md** 🎯
High-level optimization strategy:
- Problem overview
- Optimization approaches (Triton vs CUDA vs PyTorch)
- Performance targets
- Learning resources

**When to read**: When planning your implementation approach.

### 4. **submission_pytorch_optimized.py** 🐍
Intermediate optimization using pure PyTorch:
- Faster than reference implementation
- No custom CUDA required
- Good starting point before writing kernels

**When to use**: 
- Your first submission (after reference)
- Learning the problem structure
- Baseline for comparison

### 5. **submission_starter_triton.py** ⚡
Template for Triton-based custom kernel:
- Python-based GPU programming
- Easier learning curve than CUDA
- Good performance potential

**When to use**: 
- Week 1-2 when learning custom kernels
- If you want to avoid C++/CUDA complexity

---

## 🚀 Quick Start (10 Minutes)

### Step 1: Set Up Popcorn CLI
```bash
# Clone the CLI tool
git clone https://github.com/gpu-mode/popcorn-cli
cd popcorn-cli

# Follow installation instructions
# You'll need to authenticate with your Discord account
```

### Step 2: Get Reference Files
```bash
# Clone the reference kernels repo
git clone https://github.com/gpu-mode/reference-kernels
cd reference-kernels/problems/nvidia/nvfp4_gemv

# You'll find:
# - reference.py (baseline implementation)
# - task.py (problem definition)
# - template.py (starter template)
```

### Step 3: Make First Submission
```bash
# Copy reference.py content to submission.py
# Or use the optimized PyTorch version I provided

# Submit!
popcorn submit nvidia/nvfp4_gemv submission.py

# Check the leaderboard to see your result
```

---

## 📖 Recommended Reading Order

1. **ACTION_PLAN.md** - Know what to do today ✅
2. **NVFP4_GEMV_STARTER_GUIDE.md** - Understand the big picture
3. **NVFP4_TECHNICAL_REFERENCE.md** - Learn the details
4. Start with **submission_pytorch_optimized.py**
5. Graduate to **submission_starter_triton.py** when ready

---

## 🎯 Your Goals

### Week 1 (Nov 11-17): Foundation
- ✅ Setup complete and first submission made
- ✅ Understand NVFP4 format and block scaling
- ✅ Choose implementation approach (Triton recommended)
- 🎯 Target: 2-5x faster than reference

### Week 2 (Nov 18-24): Optimization
- 🚀 Custom kernel implementation
- 📊 Profile and identify bottlenecks
- 🔧 Optimize memory access patterns
- 🎯 Target: 5-10x faster than reference

### Week 3 (Nov 25-28): Final Push
- ⚡ B200-specific optimizations
- 🎨 Fine-tuning and experimentation
- 🏆 Multiple submissions and iteration
- 🎯 Target: 10-20x faster, approaching speed of light

---

## 🆘 When You Get Stuck

### Discord Community
- Channel: **#nvidia-competition** on GPU MODE Discord
- Ask questions, share learnings, get help
- See what techniques others are discussing

### Key Resources
- [NVFP4 Blog Post](https://developer.nvidia.com/blog/introducing-nvfp4-for-efficient-and-accurate-low-precision-inference/)
- [CUTLASS GitHub](https://github.com/NVIDIA/cutlass)
- [Triton Documentation](https://triton-lang.org/)
- [GPU MODE YouTube](https://www.youtube.com/@GPUMODE) - Weekly lectures

### Debugging Checklist
- [ ] Does it pass correctness tests? (Check output matches reference)
- [ ] Is memory access coalesced? (Profile with eval.py)
- [ ] Are scale factors applied correctly? (Per 16-element blocks)
- [ ] Is NVFP4 unpacking correct? (2 values per byte)

---

## 💡 Key Insights

### This Problem is Memory-Bound
The speed of light analysis shows you're limited by memory bandwidth, not compute. Therefore:
- **Optimize memory access first**
- Coalescing is critical
- Minimize CPU-GPU transfers
- Use shared memory effectively

### Start Simple, Iterate Fast
- Don't wait for the "perfect" kernel
- Submit early versions to get feedback
- Profile to find bottlenecks
- Optimize incrementally

### The Competition is Long
You have 18 days. Pace yourself:
- Week 1: Learning and foundation
- Week 2: Implementation and optimization
- Week 3: Fine-tuning and final push

---

## 📊 Understanding Performance

### Speed of Light Targets
```
M=7168, K=16384, L=1: 8.622 μs
M=4096, K=7168, L=8:  17.275 μs
M=7168, K=2048, L=4:  4.317 μs
```

### What This Means
- **8.622 μs**: Theoretical minimum based on memory bandwidth
- Your goal: Get as close as possible (within 2x is competitive)
- Current reference: Probably 50-100+ μs (need to measure)

### Competitive Performance
- **Good**: 2-3x of speed of light (17-26 μs for first benchmark)
- **Great**: 1.5-2x of speed of light (13-17 μs)
- **Excellent**: <1.5x of speed of light (<13 μs)
- **Winner**: Closest to speed of light across all benchmarks

---

## 🔥 Motivation

**Why this is worth your time**:
- 🎓 Learn cutting-edge GPU programming
- 🏆 Win NVIDIA hardware (RTX 5080, 5090, or DGX Spark!)
- 🎫 Get invited to GTC 2026 in San Jose
- 💼 Skills highly valued at Meta and other tech companies
- 🤝 Join a community of world-class developers

**Remember**: Even if you don't win, you'll learn more about GPU programming in 3 weeks than most people learn in a year!

---

## ✅ Your Immediate TODO

- [ ] Read ACTION_PLAN.md completely
- [ ] Set up Popcorn CLI (30 min)
- [ ] Make first submission (15 min)
- [ ] Read NVFP4_TECHNICAL_REFERENCE.md (1 hour)
- [ ] Choose your implementation path (Triton recommended)
- [ ] Join #nvidia-competition on Discord
- [ ] Block out time on your calendar for next 3 weeks

---

## 📝 Notes

### For Your GitHub Repo
Consider creating this structure:
```
NVFP4-Kernel/
├── README.md (this file)
├── docs/
│   ├── technical_reference.md
│   ├── learning_resources.md
│   └── optimization_log.md
├── submissions/
│   ├── v1_pytorch_optimized.py
│   ├── v2_triton_basic.py
│   └── v3_triton_optimized.py
└── experiments/
    └── profiling_results/
```

### Track Your Progress
Keep a log of:
- Each submission's performance
- What you changed
- What worked / didn't work
- Ideas to try next

This helps you learn and shows your process!

---

## 🎉 Let's Get Started!

You're ready to begin! Remember:
1. Start with ACTION_PLAN.md
2. Make your first submission today
3. Learn iteratively
4. Ask questions on Discord
5. Have fun with it!

**Your next action**: Open ACTION_PLAN.md and follow Step 1. 

Good luck! You've got this! 🚀

---

*Created: November 11, 2025*
*Competition: NVIDIA Developer Kernel Hackathon (Kernel #1)*
*Problem: NVFP4 Batched GEMV*
*Deadline: November 28, 2025*
