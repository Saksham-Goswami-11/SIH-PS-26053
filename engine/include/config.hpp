#pragma once

/**
 * KAVACH-2.5D — Engine Configuration Constants
 *
 * Zone boundaries, bin sizes, angular resolution clamping,
 * negative obstacle thresholds, and label definitions.
 */

#include <cstdint>
#include <cmath>

namespace kavach {

// ─── Zone Boundaries (meters) ────────────────────────────────────────
constexpr float NEAR_FIELD_MAX  = 10.0f;   // 0 – 10 m
constexpr float MID_FIELD_MAX   = 50.0f;   // 10 – 50 m
constexpr float FAR_FIELD_MAX   = 100.0f;  // 50 – 100 m

// ─── Spatial Resolution per zone (meters) ────────────────────────────
constexpr float NEAR_BIN_SIZE   = 0.05f;   // 5 cm
constexpr float MID_BIN_SIZE    = 0.20f;   // 20 cm
constexpr float FAR_BIN_SIZE    = 0.50f;   // 50 cm

// ─── Angular Resolution Clamping (radians) ───────────────────────────
// Δθ(r) = Δs(r) / r, clamped to [MIN_ANGULAR_RES, MAX_ANGULAR_RES]
constexpr float MIN_ANGULAR_RES = 0.5f * (M_PI / 180.0f);   // 0.5 degrees
constexpr float MAX_ANGULAR_RES = 10.0f * (M_PI / 180.0f);  // 10 degrees

// ─── Negative Obstacle Detection ─────────────────────────────────────
constexpr float DELTA_Z_THRESHOLD = 0.15f;  // meters — cells with ΔZ ≥ this are threats

// ─── Label Definitions ───────────────────────────────────────────────
enum Label : int32_t {
    LABEL_ROAD             = 0,
    LABEL_STATIC_OBSTACLE  = 1,
    LABEL_DYNAMIC_OBJECT   = 2,
    LABEL_POTHOLE_TRENCH   = 3,
    LABEL_COUNT            = 4
};

// Tie-breaking priority: higher value wins ties (hazard labels dominate)
constexpr int32_t LABEL_PRIORITY[LABEL_COUNT] = { 0, 1, 2, 3 };

// ─── Utility Functions ───────────────────────────────────────────────

/**
 * Get the spatial bin size Δs(r) for a given radius.
 */
inline float get_bin_size(float r) {
    if (r < NEAR_FIELD_MAX)  return NEAR_BIN_SIZE;
    if (r < MID_FIELD_MAX)   return MID_BIN_SIZE;
    return FAR_BIN_SIZE;
}

/**
 * Get the angular resolution Δθ(r) for a given radius.
 * Computed as Δs(r)/r, clamped to [MIN_ANGULAR_RES, MAX_ANGULAR_RES].
 */
inline float get_angular_res(float r) {
    if (r < 1e-6f) return MAX_ANGULAR_RES;  // degenerate case at origin
    float delta_theta = get_bin_size(r) / r;
    if (delta_theta < MIN_ANGULAR_RES) return MIN_ANGULAR_RES;
    if (delta_theta > MAX_ANGULAR_RES) return MAX_ANGULAR_RES;
    return delta_theta;
}

} // namespace kavach
