#pragma once

/**
 * KAVACH-2.5D — CUDA GPU Engine (Tier 1)
 *
 * GPU-accelerated polar projection using CUDA kernels.
 * Falls back to CPU engine if CUDA is unavailable at runtime.
 */

#include "grid_polar.hpp"

namespace kavach {

class GridPolarCUDA : public GridPolarEngine {
public:
    GridPolarCUDA();
    ~GridPolarCUDA() override;

    ProcessedFrame process(const Point4* points, size_t n) override;
    ComputeTier tier() const override { return ComputeTier::CUDA_TIER_1; }
    std::string info() const override;

private:
    // Device memory pointers (opaque — allocated in .cu)
    void* d_points_     = nullptr;
    void* d_bin_keys_   = nullptr;
    void* d_cell_data_  = nullptr;
    size_t allocated_n_ = 0;

    std::string device_name_;
    float device_vram_gb_ = 0.0f;

    /**
     * Allocate / reallocate device memory for N points.
     */
    void ensure_device_memory(size_t n);

    /**
     * Free all device memory.
     */
    void free_device_memory();
};

} // namespace kavach
