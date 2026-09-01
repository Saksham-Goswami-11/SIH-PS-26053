/**
 * KAVACH-2.5D — Hardware Detector
 *
 * Runtime detection of compute tier:
 *   CUDA_TIER_1 — GPU with sufficient VRAM
 *   NUMBA_TIER_2 — CPU-only C++ fallback
 *
 * Also implements the engine factory function create_engine().
 */

#include "grid_polar.hpp"
#include "grid_polar_cpu.hpp"

#ifdef KAVACH_HAS_CUDA
#include "grid_polar_cuda.cuh"
#include <cuda_runtime.h>
#endif

#include <cstdlib>
#include <cstdio>
#include <memory>
#include <string>

namespace kavach {

static constexpr float MINIMUM_VRAM_GB = 2.0f;

/**
 * Detect the best available compute tier.
 */
static ComputeTier detect_tier() {
    // Check FORCE_CPU environment override
    const char* force_cpu = std::getenv("FORCE_CPU");
    if (force_cpu && std::string(force_cpu) == "1") {
        fprintf(stderr, "[KAVACH] FORCE_CPU=1 set — using NUMBA_TIER_2 (C++ CPU)\n");
        return ComputeTier::NUMBA_TIER_2;
    }

#ifdef KAVACH_HAS_CUDA
    int device_count = 0;
    cudaError_t err = cudaGetDeviceCount(&device_count);

    if (err != cudaSuccess || device_count == 0) {
        fprintf(stderr, "[KAVACH] No CUDA devices found — using NUMBA_TIER_2\n");
        return ComputeTier::NUMBA_TIER_2;
    }

    cudaDeviceProp prop;
    cudaGetDeviceProperties(&prop, 0);
    float vram_gb = static_cast<float>(prop.totalGlobalMem) / (1024.0f * 1024.0f * 1024.0f);

    if (vram_gb < MINIMUM_VRAM_GB) {
        fprintf(stderr, "[KAVACH] GPU '%s' has %.1f GB VRAM (< %.1f GB min) — using NUMBA_TIER_2\n",
                prop.name, vram_gb, MINIMUM_VRAM_GB);
        return ComputeTier::NUMBA_TIER_2;
    }

    fprintf(stderr, "[KAVACH] GPU '%s' detected (%.1f GB VRAM) — using CUDA_TIER_1\n",
            prop.name, vram_gb);
    return ComputeTier::CUDA_TIER_1;
#else
    fprintf(stderr, "[KAVACH] Built without CUDA support — using NUMBA_TIER_2\n");
    return ComputeTier::NUMBA_TIER_2;
#endif
}

/**
 * Factory: create the best available engine instance.
 */
std::unique_ptr<GridPolarEngine> create_engine() {
    ComputeTier tier = detect_tier();

    switch (tier) {
#ifdef KAVACH_HAS_CUDA
        case ComputeTier::CUDA_TIER_1:
            return std::make_unique<GridPolarCUDA>();
#endif
        case ComputeTier::NUMBA_TIER_2:
        default:
            return std::make_unique<GridPolarCPU>();
    }
}

} // namespace kavach
