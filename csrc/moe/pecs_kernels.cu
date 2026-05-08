#include <torch/all.h>
#include <cuda_runtime.h>
#include "moe_ops.h"

// Fused CUDA kernel to perform asynchronous H2D copies directly from pinned host memory.
// Parallelized across blocks and threads for maximum PCIe utilization.
__global__ void pecs_stage_kernel(
    const int32_t* __restrict__ expert_ids,
    const void** __restrict__ src_ptrs,
    void** __restrict__ dst_ptrs,
    int num_params,
    int64_t num_int4,
    uint32_t* __restrict__ global_bitmask) {
    
    // Each block handles one expert from the candidate list (Y dimension).
    int candidate_idx = blockIdx.y;
    int param_idx = blockIdx.z;

    int32_t expert_id = expert_ids[candidate_idx];
    if (expert_id < 0) return;

    // Use a global bitmask to deduplicate copies if the same expert is requested multiple times.
    // The mask is indexed by (param_idx * total_experts_cap + expert_id).
    // Assuming a reasonable cap like 128 for total_experts_cap.
    if (blockIdx.x == 0 && threadIdx.x == 0) {
        uint32_t bit = (1u << (expert_id % 32));
        uint32_t offset = (param_idx * 128) + (expert_id / 32);
        if (atomicOr(&global_bitmask[offset], bit) & bit) {
            // Already being copied by another block, exit.
            return;
        }
    }
    __syncthreads();

    const int4* src_base = (const int4*)src_ptrs[param_idx];
    int4* dst_base = (int4*)dst_ptrs[param_idx];

    // Offsets for this expert
    const int4* src = src_base + (int64_t)expert_id * num_int4;
    int4* dst = dst_base + (int64_t)expert_id * num_int4;

    // Parallel copy using all threads in the block
    for (int64_t i = (int64_t)blockIdx.x * blockDim.x + threadIdx.x; i < num_int4; i += (int64_t)gridDim.x * blockDim.x) {
        dst[i] = src[i];
    }
}

void pecs_stage_experts(
    const torch::Tensor& expert_ids,
    const std::vector<torch::Tensor>& src_weights,
    const std::vector<torch::Tensor>& dst_buffers,
    intptr_t stream_ptr) {
    
    int num_candidates = expert_ids.numel();
    if (num_candidates == 0 || src_weights.empty()) return;
    
    cudaStream_t stream = reinterpret_cast<cudaStream_t>(stream_ptr);
    int num_params = src_weights.size();

    // Collect pointers on host
    static thread_local std::vector<const void*> h_src_ptrs;
    static thread_local std::vector<void*> h_dst_ptrs;
    h_src_ptrs.assign(num_params, nullptr);
    h_dst_ptrs.assign(num_params, nullptr);

    for (int i = 0; i < num_params; ++i) {
        h_src_ptrs[i] = src_weights[i].data_ptr();
        h_dst_ptrs[i] = dst_buffers[i].data_ptr();
    }

    // Allocate temp device buffers for pointers
    void** d_src_ptrs;
    void** d_dst_ptrs;
    cudaMallocAsync(&d_src_ptrs, num_params * sizeof(void*), stream);
    cudaMallocAsync(&d_dst_ptrs, num_params * sizeof(void*), stream);
    cudaMemcpyAsync(d_src_ptrs, h_src_ptrs.data(), num_params * sizeof(void*), cudaMemcpyHostToDevice, stream);
    cudaMemcpyAsync(d_dst_ptrs, h_dst_ptrs.data(), num_params * sizeof(void*), cudaMemcpyHostToDevice, stream);

    int64_t num_experts = src_weights[0].size(0);
    int64_t expert_bytes = (src_weights[0].numel() / num_experts) * src_weights[0].element_size();
    int64_t num_int4 = expert_bytes / sizeof(int4);

    // Allocate and zero out a bitmask for deduplication (128 bits per parameter)
    uint32_t* d_bitmask;
    cudaMallocAsync(&d_bitmask, num_params * (128 / 8), stream);
    cudaMemsetAsync(d_bitmask, 0, num_params * (128 / 8), stream);

    // Launch configuration:
    // Z dimension = parameter index (w1, w2, w3)
    // Y dimension = candidate index (0..num_candidates-1)
    // X dimension = parallel blocks per copy task
    dim3 blocks(16, num_candidates, num_params);
    dim3 threads(256);

    pecs_stage_kernel<<<blocks, threads, 0, stream>>>(
        expert_ids.data_ptr<int32_t>(),
        (const void**)d_src_ptrs,
        (void**)d_dst_ptrs,
        num_params,
        num_int4,
        d_bitmask
    );

    cudaFreeAsync(d_src_ptrs, stream);
    cudaFreeAsync(d_dst_ptrs, stream);
    cudaFreeAsync(d_bitmask, stream);
}
