/**
 * KAVACH-2.5D — CPU Fallback Engine Implementation (Tier 2)
 *
 * Full C++ implementation of the foveated 2.5D polar projection pipeline:
 *   1. Polar coordinate conversion: (X,Y) → (r,θ)
 *   2. Adaptive resolution binning by zone
 *   3. Spatial hashing into polar cells
 *   4. Cell-level aggregation: max(Z), min(Z), mode(Label)
 *   5. Negative obstacle detection via ΔZ threshold
 */

#include "grid_polar_cpu.hpp"
#include "config.hpp"

#include <cmath>
#include <chrono>
#include <unordered_map>

namespace kavach {

// ─── Spatial Hashing ─────────────────────────────────────────────────

uint64_t GridPolarCPU::compute_hash(float r, float theta) {
    float bin_size = get_bin_size(r);
    float angular_res = get_angular_res(r);

    // Compute bin indices
    auto bin_r = static_cast<int32_t>(std::floor(r / bin_size));
    // Normalize theta to [0, 2π) before binning
    if (theta < 0.0f) theta += 2.0f * static_cast<float>(M_PI);
    auto bin_theta = static_cast<int32_t>(std::floor(theta / angular_res));

    // Pack into uint64_t: upper 32 bits = bin_r, lower 32 bits = bin_theta
    return (static_cast<uint64_t>(static_cast<uint32_t>(bin_r)) << 32) |
            static_cast<uint64_t>(static_cast<uint32_t>(bin_theta));
}

// ─── Main Processing Pipeline ────────────────────────────────────────

ProcessedFrame GridPolarCPU::process(const Point4* points, size_t n) {
    auto t_start = std::chrono::high_resolution_clock::now();

    ProcessedFrame result;
    result.raw_point_count = static_cast<int>(n);

    // Phase 1 & 2: Polar conversion + spatial hashing + accumulation
    std::unordered_map<uint64_t, CellAccumulator> cell_map;
    cell_map.reserve(n / 10);  // rough estimate: ~10 points per cell average

    for (size_t i = 0; i < n; ++i) {
        const Point4& pt = points[i];

        // Polar conversion
        float r = std::sqrt(pt.x * pt.x + pt.y * pt.y);
        float theta = std::atan2(pt.y, pt.x);

        // Skip points beyond far field
        if (r > FAR_FIELD_MAX) continue;

        // Compute spatial hash key
        uint64_t key = compute_hash(r, theta);

        // Accumulate into cell
        cell_map[key].add_point(pt.x, pt.y, pt.z, pt.label);
    }

    // Phase 3: Finalize cells + extract threats
    result.cells.reserve(cell_map.size());
    result.threats.reserve(64);  // pre-alloc for common case

    for (const auto& [key, accum] : cell_map) {
        if (accum.point_count == 0) continue;

        GridCell cell = accum.to_cell();
        result.cells.push_back(cell);

        // Check for negative obstacles: ΔZ ≥ threshold
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

    auto t_end = std::chrono::high_resolution_clock::now();
    result.latency_ms = std::chrono::duration<double, std::milli>(t_end - t_start).count();

    return result;
}

} // namespace kavach
