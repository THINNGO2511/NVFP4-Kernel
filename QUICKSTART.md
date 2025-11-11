# Your First 24 Hours - Quick Start Guide

**Goal**: Get your environment set up and make your first submission to the leaderboard!

## Hour 1: Setup Popcorn CLI

### Step 1: Download Popcorn CLI
```bash
# Go to: https://github.com/gpu-mode/popcorn-cli/releases
# Download the latest release for macOS

# For example (check for latest version):
cd ~/Downloads
# If on Apple Silicon Mac:
curl -L -o popcorn-cli https://github.com/gpu-mode/popcorn-cli/releases/download/v0.x.x/popcorn-cli-darwin-arm64
# If on Intel Mac:
curl -L -o popcorn-cli https://github.com/gpu-mode/popcorn-cli/releases/download/v0.x.x/popcorn-cli-darwin-amd64

# Make it executable
chmod +x popcorn-cli

# Move to your PATH
sudo mv popcorn-cli /usr/local/bin/
```

### Step 2: Configure Popcorn CLI
```bash
# 1. Join GPU MODE Discord if you haven't
# Go to: https://discord.gg/gpumode

# 2. In Discord, go to #nvidia-competition channel and type:
/get-api-url

# 3. Copy the URL you get and export it:
export POPCORN_API_URL="the-url-you-got-from-discord"

# 4. Register with Discord (recommended):
popcorn-cli register discord

# 5. Verify it worked:
cat $HOME/.popcorn.yaml
# You should see your client ID
```

### Step 3: Test Popcorn CLI
```bash
# See available leaderboards
popcorn-cli list

# You should see 'nvfp4_gemv' in the list
```

✅ **Checkpoint**: Popcorn CLI is working and you're registered!

---

## Hour 2-3: Setup Your Dev Environment

### Clone Your Repo
```bash
cd ~/projects  # or wherever you keep code
git clone https://github.com/THINNGO2511/NVFP4-Kernel.git
cd NVFP4-Kernel
```

### Create Project Structure
```bash
# Create directories
mkdir -p docs/notes kernels/naive kernels/optimized kernels/final tests benchmarks

# Add the files we created
# (You already have README.md and learning_plan.md)
```

### Install Python Dependencies
```bash
# Create virtual environment
python3 -m venv venv
source venv/bin/activate

# Install required packages
pip install torch numpy

# For Triton (if you'll use it locally for testing)
pip install triton
```

✅ **Checkpoint**: Dev environment ready!

---

## Hour 4-6: Read and Understand

### 1. Read the Problem Description (30 min)
Go to Discord #nvidia-competition channel and find:
- Official problem description
- Reference implementation link
- Performance targets

### 2. Understand GEMV (30 min)
Read these in order:
1. Your README.md (you already have this!)
2. [Simple GEMV Explanation](https://www.bealto.com/gpu-gemv.html)

**Key takeaway**: GEMV is just computing many dot products in parallel!

### 3. Skim the NVFP4 Blog Post (1 hour)
- [NVIDIA NVFP4 Blog](https://developer.nvidia.com/blog/introducing-nvfp4-for-efficient-and-accurate-low-precision-inference/)

Don't worry if you don't understand everything yet. Focus on:
- What is NVFP4? (4-bit floating point)
- Why does it have two scales? (micro-block + tensor)
- How is it different from FP16? (much smaller!)

✅ **Checkpoint**: You understand what you're building!

---

## Hour 7-12: CUDA Crash Course

Since you have no kernel experience, you need to learn CUDA basics first.

### Watch These Videos (in order):
1. **"Introduction to CUDA" by GPU MODE** (~30 min)
   - Search YouTube: "GPU MODE CUDA introduction"

2. **"CUDA Memory Model" by GPU MODE** (~30 min)

3. **NVIDIA's "An Even Easier Introduction to CUDA"** (~20 min)
   - Search: NVIDIA CUDA introduction

### Read These (skim first, deep dive later):
1. [CUDA C Programming Guide - Chapter 2](https://docs.nvidia.com/cuda/cuda-c-programming-guide/index.html#programming-model)
   - Focus on: threads, blocks, grids
   - Don't worry about advanced features yet

### Key Concepts to Understand:
```
Grid (entire GPU)
  └── Blocks (can run in parallel on different SMs)
        └── Threads (execute same code, different data)
              └── Each thread has its own ID

Thread ID calculation:
  int tid = blockIdx.x * blockDim.x + threadIdx.x;
```

✅ **Checkpoint**: You understand the CUDA execution model!

---

## Hour 13-16: Look at Reference Implementation

### Find the Reference Code
1. Go to: https://github.com/gpu-mode/reference-kernels
2. Navigate to: `problems/nvidia/nvfp4_gemv/`
3. Look for: `reference.py` or similar

### Study the Reference (Don't Copy!)
**Your goal**: Understand the structure, not memorize the code

Look for:
- How is the kernel launched? (grid/block configuration)
- How is NVFP4 data unpacked?
- How are the scales applied?
- What does the main computation loop look like?

### Take Notes
Create `docs/notes/reference_study.md` and write:
- What you understand
- What you don't understand yet (that's okay!)
- Questions to ask in Discord

✅ **Checkpoint**: You've seen what a working implementation looks like!

---

## Hour 17-20: Your First Kernel (Simplified!)

### Don't Start with NVFP4!
Start with regular FP32 GEMV first. You'll add FP4 later.

### Option 1: Use the Template
Use `naive_gemv.py` that we created:
```bash
cd ~/NVFP4-Kernel
python naive_gemv.py
```

### Option 2: Write from Scratch
```python
# pseudo-CUDA code
__global__ void gemv_kernel(float* Y, float* A, float* X, int M, int K) {
    int m = blockIdx.x * blockDim.x + threadIdx.x;
    
    if (m < M) {
        float sum = 0.0f;
        for (int k = 0; k < K; k++) {
            sum += A[m * K + k] * X[k];
        }
        Y[m] = sum;
    }
}
```

### Test Your Kernel
```python
# Create small test case
A = random matrix of size M x K
X = random vector of size K
Y_your_kernel = your_gemv(A, X)
Y_pytorch = A @ X  # PyTorch reference

# Check if they match
assert torch.allclose(Y_your_kernel, Y_pytorch, atol=1e-5)
```

✅ **Checkpoint**: You have a working (slow) FP32 GEMV kernel!

---

## Hour 21-24: First Submission!

### Adapt for the Competition Format
Check the submission requirements:
1. What interface does the submission need? (check Discord/reference)
2. What file format? (Usually a Python file with specific function signature)
3. Any special requirements?

### Make Your Submission File
```python
# submission.py
# Follow the exact format specified in the competition rules

def nvfp4_gemv(A_fp4, scales, X, ...):
    # For now, you might convert FP4 to FP32 as a placeholder
    A_fp32 = dequantize_fp4(A_fp4, scales)  # simplified
    return your_fp32_gemv(A_fp32, X)
```

### Submit!
```bash
# Make sure you're in the right directory
cd ~/NVFP4-Kernel

# Submit to the leaderboard
popcorn-cli submit \
  --gpu B200 \
  --leaderboard nvfp4_gemv \
  --mode leaderboard \
  submission.py

# Check your position!
popcorn-cli leaderboard nvfp4_gemv
```

### Don't Worry About Performance Yet!
Your first submission will probably be slow. That's expected! The goals are:
1. ✓ Submission process works
2. ✓ Code runs without errors  
3. ✓ You're on the leaderboard
4. ✓ You have a baseline to improve from

✅ **Checkpoint**: YOU'RE IN THE COMPETITION! 🎉

---

## End of Day 1: What You've Accomplished

- [x] Environment set up ✓
- [x] Popcorn CLI working ✓
- [x] Basic CUDA knowledge ✓
- [x] Studied reference implementation ✓
- [x] Working simple kernel ✓
- [x] First submission to leaderboard ✓

---

## Tomorrow (Day 2): Next Steps

1. **Study the Performance Numbers**
   - How much slower is your kernel vs the reference?
   - What's the theoretical peak performance?

2. **Start Week 1 Learning Plan**
   - Follow the detailed learning plan (learning_plan.md)
   - Focus on memory access patterns

3. **Join Discord Discussions**
   - Ask questions in #nvidia-competition
   - Learn from what others are doing
   - Share your progress!

4. **Profile Your Kernel**
   - Where is it spending time?
   - Memory bound or compute bound?

---

## Common Issues & Solutions

### Issue: "command not found: popcorn-cli"
**Solution**: Make sure you moved it to /usr/local/bin and it's executable
```bash
which popcorn-cli  # Should show: /usr/local/bin/popcorn-cli
ls -l /usr/local/bin/popcorn-cli  # Should show: -rwxr-xr-x
```

### Issue: "No CUDA-capable device detected"
**Don't worry!** You're developing on Mac. Testing happens on remote GPUs via Popcorn CLI.

### Issue: "Import error: No module named 'triton'"
```bash
pip install triton
# Or use PyTorch's CUDA kernels instead
```

### Issue: Can't find the reference implementation
Check:
1. GPU MODE Discord #nvidia-competition channel (pinned messages)
2. https://github.com/gpu-mode/reference-kernels

### Issue: Submission rejected
- Check file format matches requirements
- Verify function signature is correct
- Make sure you're submitting to the right leaderboard

---

## Resources Quick Links

- **Discord**: https://discord.gg/gpumode → #nvidia-competition
- **Popcorn CLI**: https://github.com/gpu-mode/popcorn-cli
- **Reference Kernels**: https://github.com/gpu-mode/reference-kernels
- **NVFP4 Blog**: https://developer.nvidia.com/blog/introducing-nvfp4-for-efficient-and-accurate-low-precision-inference/
- **CUDA Guide**: https://docs.nvidia.com/cuda/cuda-c-programming-guide/

---

## Mindset for Day 1

Remember:
- **It's okay to not understand everything** - You're learning!
- **Slow is fine** - Your first submission will be slow, that's expected
- **Ask questions** - The community is helpful
- **Have fun** - This is an amazing learning opportunity!

The goal today is NOT to win the competition. The goal is to:
1. Get your environment working
2. Understand what you're building
3. Make your first submission
4. Start your learning journey

Everything else comes with time and practice. You've got 18 more days! 🚀

---

**Let's go! Time to start Hour 1!** ⏰
