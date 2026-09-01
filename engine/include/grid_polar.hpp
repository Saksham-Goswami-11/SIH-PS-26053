#pragma once

/**
 * KAVACH-2.5D — Grid Polar Engine Interface
 *
 * Abstract base class for the foveated 2.5D mapping engine.
 * Implementations: GridPolarCPU (C++ fallback) and GridPolarCUDA (GPU).
 */

#include "types.hpp"
#include <memory>
#include <string>

namespace kavach {

/**
 * Compute tier enumeration.
 */
enum class ComputeTier {
    CUDA_TIER_1,    // GPU-accelerated path
    NUMBA_TIER_2    // CPU-only C++ path (named NUMBA_TIER_2 for PRD compatibility)
};

inline std::string tier_to_string(ComputeTier tier) {
    switch (tier) {
        case ComputeTier::CUDA_TIER_1:  return "CUDA_TIER_1";
        case ComputeTier::NUMBA_TIER_2: return "NUMBA_TIER_2";
        default: return "UNKNOWN";
    }
}

/**
 * Abstract engine interface.
 */
class GridPolarEngine {
public:
    virtual ~GridPolarEngine() = default;

    /**
     * Process a frame of LiDAR points.
     *
     * @param points  Pointer to N Point4 structs (contiguous)
     * @param n       Number of points
     * @return        ProcessedFrame with compressed cells, threats, telemetry
     */
    virtual ProcessedFrame process(const Point4* points, size_t n) = 0;

    /**
     * Get the compute tier this engine instance uses.
     */
    virtual ComputeTier tier() const = 0;

    /**
     * Get human-readable engine description.
     */
    virtual std::string info() const = 0;
};

/**
 * Factory: auto-selects the best available engine tier.
 *
 * Returns CUDA_TIER_1 if a capable GPU is detected (and FORCE_CPU is not set),
 * otherwise returns NUMBA_TIER_2 (CPU).
 */
std::unique_ptr<GridPolarEngine> create_engine();

} // namespace kavach
