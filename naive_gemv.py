"""
Naive Batched GEMV Kernel - Starting Point
===========================================

This is a simple, unoptimized implementation of batched GEMV to help you understand
the operation before diving into optimizations.

GEMV Operation: Y[batch][m] = sum_k(A[batch][m][k] * X[batch][k])

Where:
- A: Matrix of shape [batch_size, M, K]
- X: Vector of shape [batch_size, K]
- Y: Output vector of shape [batch_size, M]

This implementation uses FP32 (not FP4 yet) to focus on correctness first.
"""

import torch
import triton
import triton.language as tl


@triton.jit
def naive_gemv_kernel(
    # Pointers to matrices
    Y_ptr,  # Output [batch_size, M]
    A_ptr,  # Matrix [batch_size, M, K]
    X_ptr,  # Vector [batch_size, K]
    # Matrix dimensions
    M: tl.constexpr,
    K: tl.constexpr,
    batch_size: tl.constexpr,
    # Strides for memory access
    stride_Y_batch,
    stride_Y_m,
    stride_A_batch,
    stride_A_m,
    stride_A_k,
    stride_X_batch,
    stride_X_k,
):
    """
    Naive GEMV kernel using Triton
    
    Each program instance computes one output element Y[batch, m]
    """
    # Get program ID
    pid = tl.program_id(axis=0)
    
    # Compute batch index and m index from program ID
    batch_idx = pid // M
    m_idx = pid % M
    
    # Boundary check
    if batch_idx >= batch_size or m_idx >= M:
        return
    
    # Initialize accumulator
    acc = 0.0
    
    # Compute dot product: sum over K dimension
    for k in range(K):
        # Load A[batch, m, k]
        a_ptr = A_ptr + batch_idx * stride_A_batch + m_idx * stride_A_m + k * stride_A_k
        a_val = tl.load(a_ptr)
        
        # Load X[batch, k]
        x_ptr = X_ptr + batch_idx * stride_X_batch + k * stride_X_k
        x_val = tl.load(x_ptr)
        
        # Accumulate
        acc += a_val * x_val
    
    # Store result Y[batch, m]
    y_ptr = Y_ptr + batch_idx * stride_Y_batch + m_idx * stride_Y_m
    tl.store(y_ptr, acc)


def naive_gemv_triton(A: torch.Tensor, X: torch.Tensor) -> torch.Tensor:
    """
    Wrapper function for naive GEMV kernel
    
    Args:
        A: Matrix of shape [batch_size, M, K]
        X: Vector of shape [batch_size, K]
    
    Returns:
        Y: Output vector of shape [batch_size, M]
    """
    batch_size, M, K = A.shape
    assert X.shape == (batch_size, K), f"X shape {X.shape} doesn't match expected {(batch_size, K)}"
    
    # Allocate output
    Y = torch.empty((batch_size, M), device=A.device, dtype=A.dtype)
    
    # Launch kernel
    grid = (batch_size * M,)
    
    naive_gemv_kernel[grid](
        Y, A, X,
        M=M,
        K=K,
        batch_size=batch_size,
        stride_Y_batch=Y.stride(0),
        stride_Y_m=Y.stride(1),
        stride_A_batch=A.stride(0),
        stride_A_m=A.stride(1),
        stride_A_k=A.stride(2),
        stride_X_batch=X.stride(0),
        stride_X_k=X.stride(1),
    )
    
    return Y


# ============================================================================
# Test and Validation Code
# ============================================================================

def test_naive_gemv():
    """Test the naive GEMV implementation against PyTorch reference"""
    print("Testing Naive Batched GEMV...")
    
    # Test parameters
    batch_size = 4
    M = 128
    K = 256
    
    # Create test data
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    A = torch.randn(batch_size, M, K, device=device, dtype=torch.float32)
    X = torch.randn(batch_size, K, device=device, dtype=torch.float32)
    
    # Compute with our kernel
    Y_triton = naive_gemv_triton(A, X)
    
    # Compute reference with PyTorch
    Y_torch = torch.bmm(A, X.unsqueeze(-1)).squeeze(-1)
    
    # Check correctness
    max_diff = torch.max(torch.abs(Y_triton - Y_torch)).item()
    mean_diff = torch.mean(torch.abs(Y_triton - Y_torch)).item()
    
    print(f"Max difference: {max_diff:.6f}")
    print(f"Mean difference: {mean_diff:.6f}")
    
    if max_diff < 1e-3:
        print("✓ Test PASSED!")
    else:
        print("✗ Test FAILED!")
        print(f"Expected:\n{Y_torch[0, :10]}")
        print(f"Got:\n{Y_triton[0, :10]}")
    
    return max_diff < 1e-3


def benchmark_naive_gemv():
    """Benchmark the naive GEMV implementation"""
    print("\nBenchmarking Naive Batched GEMV...")
    
    # Benchmark parameters
    batch_size = 16
    M = 4096
    K = 4096
    num_iterations = 100
    
    device = 'cuda'
    A = torch.randn(batch_size, M, K, device=device, dtype=torch.float32)
    X = torch.randn(batch_size, K, device=device, dtype=torch.float32)
    
    # Warmup
    for _ in range(10):
        Y = naive_gemv_triton(A, X)
    torch.cuda.synchronize()
    
    # Benchmark
    import time
    start = time.time()
    for _ in range(num_iterations):
        Y = naive_gemv_triton(A, X)
    torch.cuda.synchronize()
    end = time.time()
    
    avg_time_ms = (end - start) / num_iterations * 1000
    
    # Calculate FLOPs
    # Each GEMV: M * (2K - 1) ≈ 2MK FLOPs
    # Batched: batch_size * 2MK FLOPs
    flops = batch_size * M * 2 * K
    gflops = (flops / (avg_time_ms / 1000)) / 1e9
    
    print(f"Average time: {avg_time_ms:.3f} ms")
    print(f"Performance: {gflops:.2f} GFLOPS")
    print(f"Batch size: {batch_size}, M: {M}, K: {K}")
    
    return avg_time_ms


if __name__ == "__main__":
    print("=" * 60)
    print("Naive Batched GEMV - Starting Implementation")
    print("=" * 60)
    
    # Run tests
    test_passed = test_naive_gemv()
    
    if test_passed:
        # Run benchmark
        benchmark_naive_gemv()
    else:
        print("\nPlease fix the implementation before benchmarking.")
    
    print("\n" + "=" * 60)
    print("Next Steps:")
    print("1. Understand why this implementation is slow")
    print("2. Profile with NVIDIA Nsight")
    print("3. Identify memory access bottlenecks")
    print("4. Move on to optimizations!")
    print("=" * 60)
