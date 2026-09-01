#pragma once

/**
 * KAVACH-2.5D — CPU Fallback Engine (Tier 2)
 *
 * Pure C++ implementation of the foveated 2.5D polar projection.
 * Uses std::unordered_map for spatial hashing.
 */

#include "grid_polar.hpp"

namespace kavach {

class GridPolarCPU : public GridPolarEngine {
public:
    GridPolarCPU() = default;
    ~GridPolarCPU() override = default;

    ProcessedFrame process(const Point4* points, size_t n) override;
    ComputeTier tier() const override { return ComputeTier::NUMBA_TIER_2; }
    std::string info() const override { return "GridPolarCPU — C++ Tier 2 Fallback"; }

private:
    /**
     * Compute the spatial hash key for a point given its polar coordinates.
     * Packs (bin_r, bin_theta) into a single uint64_t.
     */
    static uint64_t compute_hash(float r, float theta);
};

} // namespace kavach
