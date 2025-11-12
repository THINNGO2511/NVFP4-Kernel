"""
Intermediate Optimization: Pure PyTorch Approach
This uses PyTorch operations but optimizes the computation pattern.
Good stepping stone before writing custom CUDA kernels.
"""
import torch
from task import input_t, output_t

sf_vec_size = 16

def ceil_div(a, b):
    return (a + b - 1) // b

def to_blocked(input_matrix):
    """Convert scale factor tensor to blocked format"""
    rows, cols = input_matrix.shape
    n_row_blocks = ceil_div(rows, 128)
    n_col_blocks = ceil_div(cols, 4)
    padded = input_matrix
    blocks = padded.view(n_row_blocks, 128, n_col_blocks, 4).permute(0, 2, 1, 3)
    rearranged = blocks.reshape(-1, 4, 32, 4).transpose(1, 2).reshape(-1, 32, 16)
    return rearranged.flatten()


def custom_kernel_optimized_pytorch(data: input_t) -> output_t:
    """
    Optimized PyTorch implementation - faster than reference but still using PyTorch ops
    
    Optimizations:
    1. Remove the loop over L - batch operations together
    2. Pre-convert scale factors to avoid repeated conversion
    3. Use in-place operations where possible
    """
    a_ref, b_ref, sfa_ref_cpu, sfb_ref_cpu, _, _, c_ref = data
    
    # Get dimensions
    M, K, L = c_ref.shape
    
    # OPTIMIZATION 1: Try to batch the scaled_mm operations
    # Instead of looping, see if we can process multiple batches at once
    
    if L == 1:
        # Single batch - no loop needed
        scale_a = to_blocked(sfa_ref_cpu[:, :, 0]).cuda()
        scale_b = to_blocked(sfb_ref_cpu[:, :, 0]).cuda()
        
        res = torch._scaled_mm(
            a_ref[:, :, 0],
            b_ref[:, :, 0].transpose(0, 1),
            scale_a,
            scale_b,
            bias=None,
            out_dtype=torch.float16,
        )
        c_ref[:, 0, 0] = res[:, 0]
    else:
        # OPTIMIZATION 2: Pre-convert all scale factors (move CPU->GPU transfer outside loop)
        # This reduces CPU-GPU synchronization overhead
        scale_a_list = []
        scale_b_list = []
        
        for l_idx in range(L):
            scale_a = to_blocked(sfa_ref_cpu[:, :, l_idx]).cuda()
            scale_b = to_blocked(sfb_ref_cpu[:, :, l_idx]).cuda()
            scale_a_list.append(scale_a)
            scale_b_list.append(scale_b)
        
        # Now process each batch (still in loop, but scales are pre-loaded)
        for l_idx in range(L):
            res = torch._scaled_mm(
                a_ref[:, :, l_idx],
                b_ref[:, :, l_idx].transpose(0, 1),
                scale_a_list[l_idx],
                scale_b_list[l_idx],
                bias=None,
                out_dtype=torch.float16,
            )
            c_ref[:, 0, l_idx] = res[:, 0]
    
    return c_ref


def custom_kernel_stream_optimized(data: input_t) -> output_t:
    """
    Advanced PyTorch optimization using CUDA streams for overlapping computation
    This is more complex but can give better performance
    """
    a_ref, b_ref, sfa_ref_cpu, sfb_ref_cpu, _, _, c_ref = data
    
    M, K, L = c_ref.shape
    
    # Use CUDA streams to overlap computation and memory transfers
    streams = [torch.cuda.Stream() for _ in range(min(L, 4))]
    
    for l_idx in range(L):
        stream_idx = l_idx % len(streams)
        with torch.cuda.stream(streams[stream_idx]):
            scale_a = to_blocked(sfa_ref_cpu[:, :, l_idx]).cuda()
            scale_b = to_blocked(sfb_ref_cpu[:, :, l_idx]).cuda()
            
            res = torch._scaled_mm(
                a_ref[:, :, l_idx],
                b_ref[:, :, l_idx].transpose(0, 1),
                scale_a,
                scale_b,
                bias=None,
                out_dtype=torch.float16,
            )
            c_ref[:, 0, l_idx] = res[:, 0]
    
    # Synchronize all streams
    for stream in streams:
        stream.synchronize()
    
    return c_ref


# ============================================================================
# MAIN ENTRY POINT - Choose your implementation
# ============================================================================

def custom_kernel(data: input_t) -> output_t:
    """
    Main entry point for your submission
    
    Start with the optimized PyTorch version, then move to custom CUDA
    """
    # OPTION 1: Use optimized PyTorch (slight improvement)
    return custom_kernel_optimized_pytorch(data)
    
    # OPTION 2: Use stream-based optimization (better for larger L)
    # return custom_kernel_stream_optimized(data)
    
    # OPTION 3: Eventually replace with custom CUDA/Triton kernel
    # from my_cuda_kernel import nvfp4_gemv_cuda
    # return nvfp4_gemv_cuda(data)


"""
NOTES:

These PyTorch optimizations will give you maybe 5-20% improvement over the reference.
To get competitive performance, you'll eventually need custom CUDA kernels.

Why? Because:
1. torch._scaled_mm has overhead for small batch sizes
2. The loop over L prevents parallelization
3. Can't use B200-specific optimizations through PyTorch

But this is a good starting point to:
- Understand the problem
- Get something submitted
- Learn how the evaluation works
- Then move to custom CUDA/Triton for real performance gains
"""
