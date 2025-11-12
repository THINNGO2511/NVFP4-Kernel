"""
Starter Triton implementation for NVFP4 Batched GEMV
This is a simple starting point - you'll need to optimize significantly!
"""
import torch
import triton
import triton.language as tl

# NOTE: You'll need to install triton: pip install triton

@triton.jit
def nvfp4_gemv_kernel(
    # Pointers to matrices
    a_ptr, b_ptr, c_ptr,
    sfa_ptr, sfb_ptr,  # Scale factors
    # Matrix dimensions
    M, K, L,
    # Batch index
    batch_idx,
    # Strides for tensor dimensions
    stride_am, stride_ak, stride_al,
    stride_bk, stride_bl,
    stride_cm, stride_cl,
    # Tile sizes
    BLOCK_M: tl.constexpr,
    BLOCK_K: tl.constexpr,
):
    """
    Simple GEMV kernel: c[m, batch] = sum_k(a[m, k, batch] * b[k, batch])
    
    NOTE: This is a STARTER template - not optimized yet!
    You'll need to:
    1. Handle NVFP4 format properly (currently treating as regular floats)
    2. Apply block-based scaling factors
    3. Optimize memory access patterns
    4. Use tensor cores if possible
    """
    # Get program ID
    pid_m = tl.program_id(0)
    
    # Compute row index
    m_offset = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    
    # Initialize accumulator
    acc = tl.zeros((BLOCK_M,), dtype=tl.float32)
    
    # Loop over K dimension in blocks
    for k in range(0, K, BLOCK_K):
        k_offset = k + tl.arange(0, BLOCK_K)
        
        # Load A tile: [BLOCK_M, BLOCK_K]
        a_mask = (m_offset[:, None] < M) & (k_offset[None, :] < K)
        a_ptrs = (a_ptr + 
                  m_offset[:, None] * stride_am + 
                  k_offset[None, :] * stride_ak + 
                  batch_idx * stride_al)
        
        # TODO: Need to handle NVFP4 unpacking here
        # For now, loading as float16 (WRONG but compiles)
        a = tl.load(a_ptrs, mask=a_mask, other=0.0).to(tl.float32)
        
        # Load B vector: [BLOCK_K]
        b_mask = k_offset < K
        b_ptrs = b_ptr + k_offset * stride_bk + batch_idx * stride_bl
        b = tl.load(b_ptrs, mask=b_mask, other=0.0).to(tl.float32)
        
        # TODO: Apply scaling factors (currently skipped)
        # Need to load and apply scale_a and scale_b here
        
        # Accumulate: dot product
        acc += tl.sum(a * b[None, :], axis=1)
    
    # Store result
    m_mask = m_offset < M
    c_ptrs = c_ptr + m_offset * stride_cm + batch_idx * stride_cl
    tl.store(c_ptrs, acc.to(tl.float16), mask=m_mask)


def triton_nvfp4_gemv(a, b, sfa_permuted, sfb_permuted, c):
    """
    Wrapper function to launch Triton kernel
    
    Args:
        a: [M, K, L] in NVFP4
        b: [1, K, L] in NVFP4  
        sfa_permuted: [32, 4, rest_m, 4, rest_k, L] scale factors
        sfb_permuted: [32, 4, rest_n, 4, rest_k, L] scale factors
        c: [M, 1, L] output buffer
    """
    M, K, L = a.shape
    assert b.shape == (1, K, L)
    assert c.shape == (M, 1, L)
    
    # Configuration
    BLOCK_M = 64  # Tile size for M dimension
    BLOCK_K = 64  # Tile size for K dimension
    
    # Get strides
    stride_am, stride_ak, stride_al = a.stride()
    stride_bk, stride_bl = b.stride()[1], b.stride()[2]
    stride_cm, stride_cl = c.stride()[0], c.stride()[2]
    
    # Launch kernel for each batch
    for batch_idx in range(L):
        # Grid size: number of M blocks
        grid = (triton.cdiv(M, BLOCK_M),)
        
        nvfp4_gemv_kernel[grid](
            a, b, c,
            sfa_permuted, sfb_permuted,
            M, K, L,
            batch_idx,
            stride_am, stride_ak, stride_al,
            stride_bk, stride_bl,
            stride_cm, stride_cl,
            BLOCK_M=BLOCK_M,
            BLOCK_K=BLOCK_K,
        )
    
    return c


# ============================================================================
# MAIN ENTRY POINT FOR SUBMISSION
# ============================================================================

from task import input_t, output_t

def custom_kernel(data: input_t) -> output_t:
    """
    Custom kernel implementation for NVFP4 batched GEMV
    
    This is YOUR submission function. Replace the implementation below
    with your optimized kernel.
    """
    a, b, sfa, sfb, sfa_permuted, sfb_permuted, c = data
    
    # STARTER: Use Triton kernel
    # TODO: This is incomplete and won't work correctly yet!
    # You need to:
    # 1. Handle NVFP4 format properly
    # 2. Apply scaling factors
    # 3. Optimize for B200 architecture
    
    # For now, fall back to reference implementation
    # Replace this once your Triton kernel works!
    from reference import ref_kernel
    return ref_kernel(data)
    
    # Uncomment when your Triton kernel is ready:
    # return triton_nvfp4_gemv(a, b, sfa_permuted, sfb_permuted, c)


# ============================================================================
# TESTING LOCALLY
# ============================================================================

if __name__ == "__main__":
    print("Testing NVFP4 GEMV kernel locally...")
    print("NOTE: This won't work without a CUDA GPU!")
    
    # You can add basic testing here if you have a local GPU
    # Otherwise, test via Popcorn CLI submission
