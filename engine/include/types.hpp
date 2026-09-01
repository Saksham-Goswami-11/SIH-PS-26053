#pragma once

/**
 * KAVACH-2.5D — Core Data Types
 *
 * Structs for input points, output grid cells, threats,
 * and processed frame results.
 */

#include <cstdint>
#include <vector>
#include <string>

namespace kavach {

/**
 * Input point from LiDAR / inference pipeline.
 * Matches Interface 1: (X, Y, Z, Label_ID) — shape (N, 4)
 */
struct Point4 {
    float x;        // meters, ego-centric
    float y;        // meters, ego-centric
    float z;        // meters, elevation
    int32_t label;  // 0=Road, 1=Static, 2=Dynamic, 3=Pothole/Trench
};

/**
 * Output grid cell after 2.5D compression.
 * Matches Interface 2: (Center_X, Center_Y, Max_Z, Delta_Z, Label, Radius_r) — shape (M, 6)
 */
struct GridCell {
    float center_x;     // meters — cell center X
    float center_y;     // meters — cell center Y
    float max_z;        // meters — maximum elevation in cell
    float delta_z;      // meters — vertical hazard delta (max_z - min_z)
    int32_t label;      // dominant class in cell (with hazard priority tie-break)
    float radius;       // meters — distance from sensor to cell center
};

/**
 * Detected negative obstacle / threat.
 */
struct Threat {
    std::string type;           // e.g. "NEGATIVE_OBSTACLE"
    float distance_m;           // distance from sensor
    float coordinates[2];       // [x, y] of cell center
    float depth_m;              // delta_z value
};

/**
 * Accumulator used during cell aggregation.
 * Collects all points that hash to the same polar bin.
 */
struct CellAccumulator {
    float sum_x      = 0.0f;
    float sum_y      = 0.0f;
    float max_z      = -1e30f;
    float min_z      =  1e30f;
    int32_t label_counts[4] = {0, 0, 0, 0};  // per-label vote counters
    int32_t point_count = 0;

    void add_point(float x, float y, float z, int32_t label) {
        sum_x += x;
        sum_y += y;
        if (z > max_z) max_z = z;
        if (z < min_z) min_z = z;
        if (label >= 0 && label < 4) {
            label_counts[label]++;
        }
        point_count++;
    }

    /**
     * Resolve the dominant label with hazard-priority tie-breaking.
     * Among labels with equal counts, the higher-hazard label wins.
     */
    int32_t resolve_label() const {
        int32_t best_label = 0;
        int32_t best_count = 0;
        // Iterate in priority order (low to high) so ties go to higher priority
        for (int32_t i = 0; i < 4; i++) {
            if (label_counts[i] >= best_count) {
                // >= ensures higher-priority label wins ties
                best_count = label_counts[i];
                best_label = i;
            }
        }
        return best_label;
    }

    /**
     * Finalize this accumulator into a GridCell.
     */
    GridCell to_cell() const {
        GridCell cell;
        cell.center_x = sum_x / static_cast<float>(point_count);
        cell.center_y = sum_y / static_cast<float>(point_count);
        cell.max_z    = max_z;
        cell.delta_z  = max_z - min_z;
        cell.label    = resolve_label();
        cell.radius   = std::sqrt(cell.center_x * cell.center_x +
                                  cell.center_y * cell.center_y);
        return cell;
    }
};

/**
 * Full result of processing one frame.
 */
struct ProcessedFrame {
    std::vector<GridCell> cells;      // compressed 2.5D grid cells
    std::vector<Threat>   threats;    // detected negative obstacles
    int raw_point_count   = 0;       // input N
    int compressed_count  = 0;       // output M
    double latency_ms     = 0.0;     // processing time
};

} // namespace kavach
