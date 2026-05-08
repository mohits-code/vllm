#include <torch/all.h>
#include <cuda_runtime.h>
#include "moe_ops.h"

// Fused CUDA kernel to perform asynchronous H2D copies directly from pinned host memory.
// Parallelized across blocks and threads for maximum PCIe utilization.
// Tiny kernel to unique-filter expert IDs on the GPU without host syncs.
__global__ void pecs_dedup_kernel(
    const int32_t* __restrict__ input_ids,
    int32_t* __restrict__ unique_ids,
    int32_t* __restrict__ num_unique,
    int num_candidates) {
    
    if (threadIdx.x == 0) {
        uint32_t seen_mask = 0;
        int count = 0;
        for (int i = 0; i < num_candidates; ++i) {
            int32_t id = input_ids[i];
            if (id < 0) continue;
            uint32_t bit = (1u << (id % 32));
            if (!(seen_mask & bit)) {
                unique_ids[count++] = id;
                seen_mask |= bit;
            }
        }
        *num_unique = count;
    }
}

__global__ void pecs_stage_kernel(
    const int32_t* __restrict__ expert_ids,
    const int32_t* __restrict__ num_unique_ptr,
    const void** __restrict__ src_ptrs,
    void** __restrict__ dst_ptrs,
    int num_params,
    int64_t num_int4) {
    
    int num_unique = *num_unique_ptr;
    int candidate_idx = blockIdx.y;
    int param_idx = blockIdx.z;
    
    if (candidate_idx >= num_unique) return;

    int32_t expert_id = expert_ids[candidate_idx];

    const int4* src_base = (const int4*)src_ptrs[param_idx];
    int4* dst_base = (int4*)dst_ptrs[param_idx];

    const int4* src = src_base + (int64_t)expert_id * num_int4;
    int4* dst = dst_base + (int64_t)expert_id * num_int4;

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

    // Allocate temp device buffers for deduplication
    int32_t* d_unique_ids;
    int32_t* d_num_unique;
    cudaMallocAsync(&d_unique_ids, num_candidates * sizeof(int32_t), stream);
    cudaMallocAsync(&d_num_unique, sizeof(int32_t), stream);

    // Phase 1: Deduplicate on GPU (Sync-free)
    pecs_dedup_kernel<<<1, 1, 0, stream>>>(
        expert_ids.data_ptr<int32_t>(),
        d_unique_ids,
        d_num_unique,
        num_candidates
    );

    // Collect pointers on host
    static thread_local std::vector<const void*> h_src_ptrs;
    static thread_local std::vector<void*> h_dst_ptrs;
    h_src_ptrs.assign(num_params, nullptr);
    h_dst_ptrs.assign(num_params, nullptr);

    for (int i = 0; i < num_params; ++i) {
        h_src_ptrs[i] = src_weights[i].data_ptr();
        h_dst_ptrs[i] = dst_buffers[i].data_ptr();
    }

    void** d_src_ptrs;
    void** d_dst_ptrs;
    cudaMallocAsync(&d_src_ptrs, num_params * sizeof(void*), stream);
    cudaMallocAsync(&d_dst_ptrs, num_params * sizeof(void*), stream);
    cudaMemcpyAsync(d_src_ptrs, h_src_ptrs.data(), num_params * sizeof(void*), cudaMemcpyHostToDevice, stream);
    cudaMemcpyAsync(d_dst_ptrs, h_dst_ptrs.data(), num_params * sizeof(void*), cudaMemcpyHostToDevice, stream);

    int64_t num_experts = src_weights[0].size(0);
    int64_t expert_bytes = (src_weights[0].numel() / num_experts) * src_weights[0].element_size();
    int64_t num_int4 = expert_bytes / sizeof(int4);

    // Phase 2: Parallel Staging (Only for unique experts)
    dim3 blocks(16, num_candidates, num_params);
    dim3 threads(256);

    pecs_stage_kernel<<<blocks, threads, 0, stream>>>(
        d_unique_ids,
        d_num_unique,
        (const void**)d_src_ptrs,
        (void**)d_dst_ptrs,
        num_params,
        num_int4
    );

    cudaFreeAsync(d_src_ptrs, stream);
    cudaFreeAsync(d_dst_ptrs, stream);
    cudaFreeAsync(d_unique_ids, stream);
    cudaFreeAsync(d_num_unique, stream);
}
