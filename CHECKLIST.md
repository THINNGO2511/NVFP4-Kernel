# NVFP4 Kernel Hackathon - Daily Progress Checklist

Track your progress day by day! Check off items as you complete them.

---

## 🚀 Week 1: Foundations (Nov 10-16)

### Day 1 (Nov 10) - Setup & Kickoff
- [ ] Download all starter files from outputs
- [ ] Push files to GitHub repo
- [ ] Install Popcorn CLI
- [ ] Join GPU MODE Discord
- [ ] Configure Popcorn CLI (get API URL, register)
- [ ] Verify setup: `cat ~/.popcorn.yaml`
- [ ] Create project structure (run setup.sh)
- [ ] Read START_HERE.md
- [ ] Read QUICKSTART.md Hours 1-3

**Evening reflection**: What did I learn today?

---

### Day 2 (Nov 11) - CUDA Basics
- [ ] Watch: "Introduction to CUDA" video
- [ ] Read: CUDA Programming Guide Chapter 2
- [ ] Understand: Thread hierarchy (grid → block → thread)
- [ ] Practice: Calculate thread IDs
- [ ] Read: What is GEMV? (RESOURCES.md)
- [ ] Read NVFP4 blog post (first read)
- [ ] Write: 1-page summary of GEMV operation

**Evening reflection**: Can I explain CUDA thread hierarchy?

---

### Day 3 (Nov 12) - First CUDA Code
- [ ] Write: "Hello World" CUDA kernel
- [ ] Write: Vector addition kernel
- [ ] Test: Vector addition on small arrays
- [ ] Practice: Thread indexing with different block sizes
- [ ] Read: Memory hierarchy basics
- [ ] Understand: Global vs Shared vs Register memory

**Evening reflection**: Can I write a simple kernel from scratch?

---

### Day 4 (Nov 13) - Vector Operations
- [ ] Write: Vector dot product kernel (naive)
- [ ] Implement: Parallel reduction
- [ ] Test: Dot product correctness
- [ ] Compare: Your kernel vs PyTorch
- [ ] Read: Memory coalescing concepts
- [ ] Watch: GPU MODE memory optimization lecture

**Evening reflection**: Why is memory coalescing important?

---

### Day 5 (Nov 14) - Memory Deep Dive
- [ ] Read: CUDA Best Practices Guide Chapter 9
- [ ] Implement: Matrix transpose (naive)
- [ ] Implement: Matrix transpose (optimized with coalescing)
- [ ] Compare: Performance difference
- [ ] Profile: Use timing to measure bandwidth
- [ ] Understand: Why GEMV is memory-bound

**Evening reflection**: Can I explain coalesced vs strided access?

---

### Day 6 (Nov 15) - Shared Memory
- [ ] Read: "Using Shared Memory in CUDA"
- [ ] Rewrite: Dot product using shared memory
- [ ] Implement: Shared memory reduction
- [ ] Test: Verify correctness
- [ ] Measure: Performance improvement
- [ ] Read about: Bank conflicts

**Evening reflection**: When should I use shared memory?

---

### Day 7 (Nov 16) - First GEMV!
- [ ] Study: Reference implementation structure
- [ ] Implement: Naive FP32 GEMV kernel
- [ ] Test: Small matrices (verify correctness)
- [ ] Test: Against PyTorch bmm
- [ ] Measure: Baseline performance
- [ ] Submit: First version to leaderboard! 🎉
- [ ] Review: Week 1 accomplishments

**Evening reflection**: What was hardest? What's next?

---

## 💪 Week 2: Optimization (Nov 17-23)

### Day 8 (Nov 17) - Memory Coalescing
- [ ] Profile: Naive kernel with Nsight
- [ ] Identify: Memory access patterns
- [ ] Study: FastGEMV memory access
- [ ] Redesign: Kernel for coalesced access
- [ ] Implement: Coalesced version
- [ ] Measure: Improvement vs naive
- [ ] Submit: Coalesced version

**Evening reflection**: How much faster?

---

### Day 9 (Nov 18) - Vectorization
- [ ] Read: Vectorized memory access patterns
- [ ] Implement: float2 vectorized loads
- [ ] Implement: float4 vectorized loads
- [ ] Test: Correctness with vectorization
- [ ] Measure: Performance improvement
- [ ] Submit: Vectorized version
- [ ] Compare: All versions so far

**Evening reflection**: Why does vectorization help?

---

### Day 10 (Nov 19) - Shared Memory Optimization
- [ ] Design: Shared memory strategy for GEMV
- [ ] Implement: Cache input vector in shared memory
- [ ] Handle: Bank conflict avoidance
- [ ] Test: Correctness
- [ ] Measure: Performance gain
- [ ] Profile: Shared memory usage
- [ ] Submit: Shared memory version

**Evening reflection**: Is shared memory the bottleneck?

---

### Day 11 (Nov 20) - Warp-Level Optimization
- [ ] Read: Warp-level primitives
- [ ] Implement: Warp shuffle for reductions
- [ ] Test: Warp-optimized version
- [ ] Measure: Performance
- [ ] Experiment: Different warp strategies
- [ ] Submit: Best warp-optimized version

**Evening reflection**: How do warps help GEMV?

---

### Day 12 (Nov 21) - NVFP4 Implementation Start
- [ ] Re-read: NVFP4 blog post (detailed)
- [ ] Study: Reference FP4 dequantization
- [ ] Understand: Bit packing (2 values per byte)
- [ ] Implement: FP4 unpacking function
- [ ] Test: Unpacking correctness
- [ ] Implement: FP8 scale application

**Evening reflection**: Do I understand NVFP4 layout?

---

### Day 13 (Nov 22) - NVFP4 Integration
- [ ] Implement: Full NVFP4 GEMV kernel
- [ ] Apply: Both FP8 and FP32 scales
- [ ] Test: Correctness vs FP32 reference
- [ ] Debug: Any numerical issues
- [ ] Measure: Performance with FP4
- [ ] Submit: NVFP4 version
- [ ] Review: Week 2 progress

**Evening reflection**: Is NVFP4 working correctly?

---

### Day 14 (Nov 23) - Benchmark Day
- [ ] Submit: All versions to leaderboard
- [ ] Profile: Each version with Nsight
- [ ] Compare: Performance metrics
- [ ] Identify: Current bottleneck
- [ ] Analyze: Where is most time spent?
- [ ] Plan: Next optimizations
- [ ] Check: Leaderboard position

**Evening reflection**: What's limiting performance now?

---

## 🏆 Week 3: Polish & Submit (Nov 24-28)

### Day 15 (Nov 24) - Advanced Opt 1
- [ ] Choose: Optimization based on profiling
- [ ] Research: How others solved this bottleneck
- [ ] Implement: Advanced optimization 1
- [ ] Test: Correctness
- [ ] Measure: Performance gain
- [ ] Submit: If improvement found

**Evening reflection**: Did this optimization help?

---

### Day 16 (Nov 25) - Advanced Opt 2
- [ ] Implement: Advanced optimization 2
- [ ] Try: Different thread block sizes
- [ ] Experiment: Multiple elements per thread
- [ ] Test: All variations
- [ ] Measure: Best configuration
- [ ] Submit: Best version

**Evening reflection**: What configuration works best?

---

### Day 17 (Nov 26) - Final Testing
- [ ] Test: Edge cases (small/large matrices)
- [ ] Test: Different batch sizes
- [ ] Test: Non-power-of-2 dimensions
- [ ] Verify: No NaN/Inf errors
- [ ] Check: Numerical accuracy
- [ ] Profile: Final version one more time

**Evening reflection**: Are there any bugs?

---

### Day 18 (Nov 27) - Final Submission Prep
- [ ] Review: All code for clarity
- [ ] Add: Comments and documentation
- [ ] Clean: Remove debug code
- [ ] Test: Final version thoroughly
- [ ] Benchmark: Final performance numbers
- [ ] Create: Backup of best version

**Evening reflection**: Am I ready for final submission?

---

### Day 19 (Nov 28) - DEADLINE DAY! 🎯
- [ ] Morning: Final testing
- [ ] Check: Submission format is correct
- [ ] Submit: Final version to leaderboard
- [ ] Verify: Submission processed successfully
- [ ] Backup: Save final submission
- [ ] Celebrate: You completed the challenge! 🎉
- [ ] Reflect: What did I learn?

**Evening reflection**: What would I do differently next time?

---

## 📊 Progress Metrics

Track these throughout:

### Performance (vs naive baseline)
- Week 1 end: ____ x baseline
- Week 2 end: ____ x baseline  
- Week 3 end: ____ x baseline

### Leaderboard Position
- Day 7: Position ____ / ____
- Day 14: Position ____ / ____
- Day 18: Position ____ / ____
- Final: Position ____ / ____

### Learning Goals
- [ ] Understand CUDA programming model
- [ ] Can write basic kernels
- [ ] Understand memory hierarchy
- [ ] Can optimize memory access
- [ ] Understand NVFP4 format
- [ ] Can profile and benchmark
- [ ] Ready for next kernel challenge

---

## 🎯 Weekly Goals Summary

### Week 1 Success Criteria
- [ ] Environment fully working
- [ ] Naive GEMV kernel running
- [ ] First submission made
- [ ] Understand CUDA basics

### Week 2 Success Criteria
- [ ] 3-5x faster than naive
- [ ] NVFP4 format working
- [ ] Top 50% on leaderboard
- [ ] Profiling regularly

### Week 3 Success Criteria
- [ ] Advanced optimizations applied
- [ ] Top 25% on leaderboard (stretch: top 10%)
- [ ] Final submission made
- [ ] Learned GPU optimization!

---

## 📝 Daily Routine

Each day should include:

**Morning (2 hours)**
- [ ] Read documentation
- [ ] Watch video tutorials
- [ ] Plan today's coding

**Afternoon (3-4 hours)**
- [ ] Hands-on coding
- [ ] Implement features
- [ ] Test and debug

**Evening (1-2 hours)**
- [ ] Benchmark and profile
- [ ] Document learnings
- [ ] Plan tomorrow

**Before bed**
- [ ] Update this checklist
- [ ] Reflect on progress
- [ ] Review what's next

---

## 🤔 Daily Reflection Questions

Answer these each evening:

1. **What did I learn today?**
   
2. **What challenged me most?**
   
3. **What worked well?**
   
4. **What will I do differently tomorrow?**
   
5. **Am I on track with the learning plan?**

---

## 🆘 When You're Stuck

If blocked for >1 hour on same issue:

- [ ] Read relevant section in RESOURCES.md
- [ ] Search Discord for similar questions
- [ ] Ask question in Discord
- [ ] Take a break and come back
- [ ] Try a simpler approach
- [ ] Move on and return later

Remember: Getting stuck is part of learning!

---

## 🎊 Celebration Milestones

Celebrate these achievements:

- [ ] First successful kernel launch
- [ ] First submission to leaderboard
- [ ] First performance improvement
- [ ] Breaking 2x speedup
- [ ] Breaking 5x speedup
- [ ] NVFP4 format working
- [ ] Making top 50%
- [ ] Making top 25%
- [ ] Final submission complete

Each milestone deserves recognition! 🎉

---

## 📈 Final Stats

Complete on Day 19:

**Total time invested**: _____ hours

**Lines of code written**: _____ lines

**Submissions made**: _____ submissions

**Final speedup vs naive**: _____ x

**Final leaderboard position**: _____ / _____

**Most valuable lesson learned**:

**What I'm most proud of**:

**What I'll do better next time**:

---

**Print this checklist and keep it handy!**
**Update daily to track your amazing progress!**

🚀 Let's build something awesome! 🚀

---

*Created: November 10, 2025*
*Deadline: November 28, 2025*
*Days Available: 18*
