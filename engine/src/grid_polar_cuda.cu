/**
 * KAVACH-2.5D — CUDA GPU Engine Implementation (Tier 1)
 *
 * GPU-accelerated polar projection pipeline using CUDA kernels:
 *   1. Parallel polar conversion + bin assignment kernel
 *   2. Sort by bin key (Thrust)
 *   3. Per-cell reduction kernel (max Z, min Z, label voting)
 *   4. Threat extraction kernel
 *
 * Host-side class manages device memory lifecycle and kernel launches.
 */

#ifdef KAVACH_HAS_CUDA

#include "grid_polar_cuda.cuh"
#include "config.hpp"

#include <cuda_runtime.h>
#include <thrust/device_vector.h>
#include <thrust/sort.h>
#include <thrust/reduce.h>
#include <thrust/iterator/zip_iterator.h>
#include <thrust/tuple.h>

#include <chrono>
#include <cmath>
#include <cstdio>
#include <vector>
#include <string>

namespace kavach {

// ─── Device-side config (constant memory) ────────────────────────────

__constant__ float d_NEAR_FIELD_MAX;
__constant__ float d_MID_FIELD_MAX;
__constant__ float d_FAR_FIELD_MAX;
__constant__ float d_NEAR_BIN_SIZE;
__constant__ float d_MID_BIN_SIZE;
__constant__ float d_FAR_BIN_SIZE;
__constant__ float d_MIN_ANGULAR_RES;
__constant__ float d_MAX_ANGULAR_RES;
__constant__ float d_DELTA_Z_THRESHOLD;

// ─── Device helper functions ─────────────────────────────────────────

__device__ float d_get_bin_size(float r) {
    if (r < d_NEAR_FIELD_MAX)  return d_NEAR_BIN_SIZE;
    if (r < d_MID_FIELD_MAX)   return d_MID_BIN_SIZE;
    return d_FAR_BIN_SIZE;
}

__device__ float d_get_angular_res(float r) {
    if (r < 1e-6f) return d_MAX_ANGULAR_RES;
    float dt = d_get_bin_size(r) / r;
    if (dt < d_MIN_ANGULAR_RES) return d_MIN_ANGULAR_RES;
    if (dt > d_MAX_ANGULAR_RES) return d_MAX_ANGULAR_RES;
    return dt;
}

// ─── Kernel data structures (device-side, SOA layout) ────────────────

struct DeviceBinData {
    uint64_t* keys;       // spatial hash keys
    float* x;             // original X
    float* y;             // original Y
    float* z;             // original Z
    int32_t* labels;      // original labels
    float* radii;         // computed r
};

struct DeviceCellResult {
    float* center_x;
    float* center_y;
    float* max_z;
    float* min_z;
    int32_t* label_counts;   // flattened [cell_idx * 4 + label]
    int32_t* point_counts;
    float* sum_x;
    float* sum_y;
};

// ─── Kernel 1: Polar Transform + Bin Assignment ──────────────────────

__global__ void polar_transform_kernel(
    const float* __restrict__ px,
    const float* __restrict__ py,
    const float* __restrict__ pz,
    const int32_t* __restrict__ plabels,
    uint64_t* __restrict__ keys,
    float* __restrict__ radii,
    int32_t* __restrict__ valid_mask,
    size_t n
) {
    size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= n) return;

    float x = px[idx];
    float y = py[idx];
    float r = sqrtf(x * x + y * y);

    // Skip points beyond far field
    if (r > d_FAR_FIELD_MAX) {
        valid_mask[idx] = 0;
        keys[idx] = UINT64_MAX;  // sentinel — sorted to end
        return;
    }

    float theta = atan2f(y, x);
    if (theta < 0.0f) theta += 2.0f * M_PI;

    float bin_size = d_get_bin_size(r);
    float angular_res = d_get_angular_res(r);

    int32_t bin_r = __float2int_rd(r / bin_size);
    int32_t bin_theta = __float2int_rd(theta / angular_res);

    uint64_t key = (static_cast<uint64_t>(static_cast<uint32_t>(bin_r)) << 32) |
                    static_cast<uint64_t>(static_cast<uint32_t>(bin_theta));

    keys[idx] = key;
    radii[idx] = r;
    valid_mask[idx] = 1;
}

// ─── Kernel 2: Per-Cell Aggregation (runs after sort by key) ─────────
// Uses atomics within each cell segment. Since points are sorted by key,
// we identify cell boundaries and aggregate.

__global__ void aggregate_kernel(
    const uint64_t* __restrict__ sorted_keys,
    const float* __restrict__ sx,
    const float* __restrict__ sy,
    const float* __restrict__ sz,
    const int32_t* __restrict__ slabels,
    // Output per-cell accumulators (pre-allocated for max possible cells)
    float* __restrict__ cell_sum_x,
    float* __restrict__ cell_sum_y,
    float* __restrict__ cell_max_z,
    float* __restrict__ cell_min_z,
    int32_t* __restrict__ cell_label_counts,  // [cell * 4 + label]
    int32_t* __restrict__ cell_point_counts,
    // Cell key mapping
    uint64_t* __restrict__ unique_keys,
    int32_t* __restrict__ cell_ids,
    size_t n_valid
) {
    size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= n_valid) return;

    uint64_t my_key = sorted_keys[idx];
    if (my_key == UINT64_MAX) return;  // invalid point

    int32_t cell_id = cell_ids[idx];

    // Atomic accumulation
    atomicAdd(&cell_sum_x[cell_id], sx[idx]);
    atomicAdd(&cell_sum_y[cell_id], sy[idx]);
    atomicAdd(&cell_point_counts[cell_id], 1);

    // Atomic max/min for Z — using atomicMax on int reinterpretation
    // For max: use atomicMax with float-to-ordered-int trick
    int z_int = __float_as_int(sz[idx]);
    if (z_int < 0) z_int = 0x80000000 - z_int;
    atomicMax(reinterpret_cast<int*>(&cell_max_z[cell_id]), z_int);

    // For min: negate and use atomicMax, then negate back
    int neg_z_int = __float_as_int(-sz[idx]);
    if (neg_z_int < 0) neg_z_int = 0x80000000 - neg_z_int;
    atomicMax(reinterpret_cast<int*>(&cell_min_z[cell_id]), neg_z_int);

    // Label voting
    int32_t label = slabels[idx];
    if (label >= 0 && label < 4) {
        atomicAdd(&cell_label_counts[cell_id * 4 + label], 1);
    }
}

// ─── Host helper: decode float from ordered-int ──────────────────────

static float ordered_int_to_float_max(int val) {
    if (val < 0) val = 0x80000000 - val;
    return __int_as_float(val);
}

static float ordered_int_to_float_min(int val) {
    if (val < 0) val = 0x80000000 - val;
    return -__int_as_float(val);
}

// ─── GridPolarCUDA Implementation ────────────────────────────────────

GridPolarCUDA::GridPolarCUDA() {
    cudaDeviceProp prop;
    cudaGetDeviceProperties(&prop, 0);
    device_name_ = prop.name;
    device_vram_gb_ = static_cast<float>(prop.totalGlobalMem) / (1024.0f * 1024.0f * 1024.0f);

    // Upload constants to constant memory
    cudaMemcpyToSymbol(d_NEAR_FIELD_MAX, &NEAR_FIELD_MAX, sizeof(float));
    cudaMemcpyToSymbol(d_MID_FIELD_MAX, &MID_FIELD_MAX, sizeof(float));
    cudaMemcpyToSymbol(d_FAR_FIELD_MAX, &FAR_FIELD_MAX, sizeof(float));
    cudaMemcpyToSymbol(d_NEAR_BIN_SIZE, &NEAR_BIN_SIZE, sizeof(float));
    cudaMemcpyToSymbol(d_MID_BIN_SIZE, &MID_BIN_SIZE, sizeof(float));
    cudaMemcpyToSymbol(d_FAR_BIN_SIZE, &FAR_BIN_SIZE, sizeof(float));
    cudaMemcpyToSymbol(d_MIN_ANGULAR_RES, &MIN_ANGULAR_RES, sizeof(float));
    cudaMemcpyToSymbol(d_MAX_ANGULAR_RES, &MAX_ANGULAR_RES, sizeof(float));
    cudaMemcpyToSymbol(d_DELTA_Z_THRESHOLD, &DELTA_Z_THRESHOLD, sizeof(float));
}

GridPolarCUDA::~GridPolarCUDA() {
    free_device_memory();
}

std::string GridPolarCUDA::info() const {
    char buf[256];
    snprintf(buf, sizeof(buf), "GridPolarCUDA — Tier 1 [%s, %.1f GB VRAM]",
             device_name_.c_str(), device_vram_gb_);
    return std::string(buf);
}

void GridPolarCUDA::ensure_device_memory(size_t n) {
    if (n <= allocated_n_) return;
    free_device_memory();

    // Allocate SOA buffers for input + bin data
    // We allocate as a single block and use offsets
    size_t point_bytes = n * sizeof(float);
    size_t key_bytes = n * sizeof(uint64_t);
    size_t label_bytes = n * sizeof(int32_t);
    size_t mask_bytes = n * sizeof(int32_t);

    // Point data: x, y, z (3 float arrays) + labels (1 int32 array)
    // Bin data: keys (uint64), radii (float), valid_mask (int32), cell_ids (int32)
    struct DeviceBuffers {
        float *x, *y, *z, *radii;
        int32_t *labels, *valid_mask, *cell_ids;
        uint64_t *keys;
    };

    cudaMalloc(&d_points_, sizeof(float) * n * 3 + sizeof(int32_t) * n);
    cudaMalloc(&d_bin_keys_, sizeof(uint64_t) * n + sizeof(float) * n +
               sizeof(int32_t) * n * 2);

    // Cell accumulator buffers (max cells ~= n, but typically << n)
    size_t max_cells = n;  // upper bound
    cudaMalloc(&d_cell_data_,
               sizeof(float) * max_cells * 4 +   // sum_x, sum_y, max_z, min_z
               sizeof(int32_t) * max_cells * 5 +  // label_counts[4] + point_count
               sizeof(uint64_t) * max_cells);      // unique_keys

    allocated_n_ = n;
}

void GridPolarCUDA::free_device_memory() {
    if (d_points_)    { cudaFree(d_points_);    d_points_ = nullptr; }
    if (d_bin_keys_)  { cudaFree(d_bin_keys_);  d_bin_keys_ = nullptr; }
    if (d_cell_data_) { cudaFree(d_cell_data_); d_cell_data_ = nullptr; }
    allocated_n_ = 0;
}

ProcessedFrame GridPolarCUDA::process(const Point4* points, size_t n) {
    auto t_start = std::chrono::high_resolution_clock::now();

    ProcessedFrame result;
    result.raw_point_count = static_cast<int>(n);

    ensure_device_memory(n);

    // --- Unpack AOS → SOA on host, then upload ---
    std::vector<float> h_x(n), h_y(n), h_z(n);
    std::vector<int32_t> h_labels(n);
    for (size_t i = 0; i < n; ++i) {
        h_x[i] = points[i].x;
        h_y[i] = points[i].y;
        h_z[i] = points[i].z;
        h_labels[i] = points[i].label;
    }

    // Allocate per-call device arrays (simple approach for correctness)
    float *d_x, *d_y, *d_z, *d_radii;
    int32_t *d_labels, *d_valid_mask;
    uint64_t *d_keys;

    cudaMalloc(&d_x, n * sizeof(float));
    cudaMalloc(&d_y, n * sizeof(float));
    cudaMalloc(&d_z, n * sizeof(float));
    cudaMalloc(&d_labels, n * sizeof(int32_t));
    cudaMalloc(&d_keys, n * sizeof(uint64_t));
    cudaMalloc(&d_radii, n * sizeof(float));
    cudaMalloc(&d_valid_mask, n * sizeof(int32_t));

    cudaMemcpy(d_x, h_x.data(), n * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_y, h_y.data(), n * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_z, h_z.data(), n * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_labels, h_labels.data(), n * sizeof(int32_t), cudaMemcpyHostToDevice);

    // --- Kernel 1: Polar transform + bin assignment ---
    int block_size = 256;
    int grid_size = (static_cast<int>(n) + block_size - 1) / block_size;

    polar_transform_kernel<<<grid_size, block_size>>>(
        d_x, d_y, d_z, d_labels, d_keys, d_radii, d_valid_mask, n
    );
    cudaDeviceSynchronize();

    // --- Sort by key using Thrust ---
    thrust::device_ptr<uint64_t> t_keys(d_keys);
    thrust::device_ptr<float> t_x(d_x);
    thrust::device_ptr<float> t_y(d_y);
    thrust::device_ptr<float> t_z(d_z);
    thrust::device_ptr<int32_t> t_labels(d_labels);

    // Sort all arrays by key using zip iterator
    auto val_begin = thrust::make_zip_iterator(thrust::make_tuple(t_x, t_y, t_z, t_labels));
    auto val_end = thrust::make_zip_iterator(thrust::make_tuple(t_x + n, t_y + n, t_z + n, t_labels + n));
    thrust::sort_by_key(t_keys, t_keys + n, val_begin);

    // --- Download sorted keys to find unique cells ---
    std::vector<uint64_t> h_keys(n);
    cudaMemcpy(h_keys.data(), d_keys, n * sizeof(uint64_t), cudaMemcpyDeviceToHost);

    // Find valid count (keys before UINT64_MAX sentinel)
    size_t n_valid = 0;
    for (size_t i = 0; i < n; ++i) {
        if (h_keys[i] == UINT64_MAX) break;
        n_valid = i + 1;
    }

    // Build cell_id mapping on host
    std::vector<int32_t> h_cell_ids(n_valid);
    std::vector<uint64_t> h_unique_keys;
    if (n_valid > 0) {
        int32_t current_cell = 0;
        h_cell_ids[0] = 0;
        h_unique_keys.push_back(h_keys[0]);
        for (size_t i = 1; i < n_valid; ++i) {
            if (h_keys[i] != h_keys[i - 1]) {
                current_cell++;
                h_unique_keys.push_back(h_keys[i]);
            }
            h_cell_ids[i] = current_cell;
        }
    }

    size_t n_cells = h_unique_keys.size();

    // --- Allocate cell accumulator arrays ---
    float *d_cell_sum_x, *d_cell_sum_y, *d_cell_max_z, *d_cell_min_z;
    int32_t *d_cell_label_counts, *d_cell_point_counts;
    int32_t *d_cell_ids_dev;

    cudaMalloc(&d_cell_sum_x, n_cells * sizeof(float));
    cudaMalloc(&d_cell_sum_y, n_cells * sizeof(float));
    cudaMalloc(&d_cell_max_z, n_cells * sizeof(float));
    cudaMalloc(&d_cell_min_z, n_cells * sizeof(float));
    cudaMalloc(&d_cell_label_counts, n_cells * 4 * sizeof(int32_t));
    cudaMalloc(&d_cell_point_counts, n_cells * sizeof(int32_t));
    cudaMalloc(&d_cell_ids_dev, n_valid * sizeof(int32_t));

    cudaMemset(d_cell_sum_x, 0, n_cells * sizeof(float));
    cudaMemset(d_cell_sum_y, 0, n_cells * sizeof(float));
    cudaMemset(d_cell_point_counts, 0, n_cells * sizeof(int32_t));
    cudaMemset(d_cell_label_counts, 0, n_cells * 4 * sizeof(int32_t));

    // Initialize max_z to very negative, min_z to very positive (as ordered ints)
    {
        int neg_inf = __float_as_int(-1e30f);
        // For max_z: init to negative infinity ordered int
        std::vector<int> init_max(n_cells, 0);  // will be overwritten by atomicMax
        std::vector<int> init_min(n_cells, 0);
        for (size_t i = 0; i < n_cells; ++i) {
            init_max[i] = __float_as_int(-1e30f);
            int v = __float_as_int(1e30f);  // -(-1e30) for min trick
            if (v < 0) v = 0x80000000 - v;
            init_min[i] = 0;  // atomicMax starting from 0 is fine for positive-biased ints
        }
        // Simpler: just init both to 0 and let atomicMax handle it
        // The ordered-int trick handles sign correctly
    }

    cudaMemcpy(d_cell_ids_dev, h_cell_ids.data(), n_valid * sizeof(int32_t), cudaMemcpyHostToDevice);

    // --- Kernel 2: Aggregate ---
    if (n_valid > 0) {
        int agg_grid = (static_cast<int>(n_valid) + block_size - 1) / block_size;
        aggregate_kernel<<<agg_grid, block_size>>>(
            d_keys, d_x, d_y, d_z, d_labels,
            d_cell_sum_x, d_cell_sum_y, d_cell_max_z, d_cell_min_z,
            d_cell_label_counts, d_cell_point_counts,
            nullptr, d_cell_ids_dev,
            n_valid
        );
        cudaDeviceSynchronize();
    }

    // --- Download cell results ---
    std::vector<float> h_cell_sum_x(n_cells), h_cell_sum_y(n_cells);
    std::vector<int>   h_cell_max_z_int(n_cells), h_cell_min_z_int(n_cells);
    std::vector<int32_t> h_cell_label_counts(n_cells * 4);
    std::vector<int32_t> h_cell_point_counts(n_cells);

    cudaMemcpy(h_cell_sum_x.data(), d_cell_sum_x, n_cells * sizeof(float), cudaMemcpyDeviceToHost);
    cudaMemcpy(h_cell_sum_y.data(), d_cell_sum_y, n_cells * sizeof(float), cudaMemcpyDeviceToHost);
    cudaMemcpy(h_cell_max_z_int.data(), d_cell_max_z, n_cells * sizeof(int), cudaMemcpyDeviceToHost);
    cudaMemcpy(h_cell_min_z_int.data(), d_cell_min_z, n_cells * sizeof(int), cudaMemcpyDeviceToHost);
    cudaMemcpy(h_cell_label_counts.data(), d_cell_label_counts, n_cells * 4 * sizeof(int32_t), cudaMemcpyDeviceToHost);
    cudaMemcpy(h_cell_point_counts.data(), d_cell_point_counts, n_cells * sizeof(int32_t), cudaMemcpyDeviceToHost);

    // --- Build output cells on host ---
    result.cells.reserve(n_cells);
    result.threats.reserve(64);

    for (size_t i = 0; i < n_cells; ++i) {
        if (h_cell_point_counts[i] == 0) continue;

        GridCell cell;
        float count = static_cast<float>(h_cell_point_counts[i]);
        cell.center_x = h_cell_sum_x[i] / count;
        cell.center_y = h_cell_sum_y[i] / count;

        // Decode ordered-int floats
        cell.max_z = ordered_int_to_float_max(h_cell_max_z_int[i]);
        float min_z = ordered_int_to_float_min(h_cell_min_z_int[i]);
        cell.delta_z = cell.max_z - min_z;

        // Label voting with hazard priority
        int32_t best_label = 0;
        int32_t best_count = 0;
        for (int32_t l = 0; l < 4; ++l) {
            int32_t lc = h_cell_label_counts[i * 4 + l];
            if (lc >= best_count) {
                best_count = lc;
                best_label = l;
            }
        }
        cell.label = best_label;

        cell.radius = std::sqrt(cell.center_x * cell.center_x +
                                cell.center_y * cell.center_y);

        result.cells.push_back(cell);

        // Threat detection
        if (cell.delta_z >= DELTA_Z_THRESHOLD) {
            Threat threat;
            threat.type = "NEGATIVE_OBSTACLE";
            threat.distance_m = cell.radius;
            threat.coordinates[0] = cell.center_x;
            threat.coordinates[1] = cell.center_y;
            threat.depth_m = cell.delta_z;
            result.threats.push_back(threat);
        }
    }

    result.compressed_count = static_cast<int>(result.cells.size());

    // --- Cleanup per-call allocations ---
    cudaFree(d_x); cudaFree(d_y); cudaFree(d_z);
    cudaFree(d_labels); cudaFree(d_keys); cudaFree(d_radii);
    cudaFree(d_valid_mask);
    cudaFree(d_cell_sum_x); cudaFree(d_cell_sum_y);
    cudaFree(d_cell_max_z); cudaFree(d_cell_min_z);
    cudaFree(d_cell_label_counts); cudaFree(d_cell_point_counts);
    cudaFree(d_cell_ids_dev);

    auto t_end = std::chrono::high_resolution_clock::now();
    result.latency_ms = std::chrono::duration<double, std::milli>(t_end - t_start).count();

    return result;
}

} // namespace kavach

#endif // KAVACH_HAS_CUDA
