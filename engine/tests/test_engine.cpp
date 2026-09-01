/**
 * KAVACH-2.5D — Engine Unit Tests
 *
 * Basic tests for the polar projection engine.
 * Validates compression ratio, output shape, ΔZ detection, and label tie-breaking.
 */

#include "grid_polar.hpp"
#include "config.hpp"
#include "types.hpp"

#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cassert>
#include <vector>
#include <random>
#include <chrono>

using namespace kavach;

// ─── Test Helpers ────────────────────────────────────────────────────

static std::vector<Point4> generate_test_points(size_t n, unsigned seed = 42) {
    std::mt19937 rng(seed);
    std::uniform_real_distribution<float> xy_dist(-80.0f, 80.0f);
    std::uniform_real_distribution<float> z_dist(-1.0f, 3.0f);
    std::uniform_int_distribution<int32_t> label_dist(0, 3);

    std::vector<Point4> points(n);
    for (size_t i = 0; i < n; ++i) {
        points[i].x = xy_dist(rng);
        points[i].y = xy_dist(rng);
        points[i].z = z_dist(rng);
        points[i].label = label_dist(rng);
    }
    return points;
}

static void add_pothole(std::vector<Point4>& points, float cx, float cy, float depth, int count) {
    std::mt19937 rng(123);
    std::uniform_real_distribution<float> jitter(-0.02f, 0.02f);
    for (int i = 0; i < count; ++i) {
        Point4 pt;
        pt.x = cx + jitter(rng);
        pt.y = cy + jitter(rng);
        pt.z = -depth + jitter(rng) * 0.1f;
        pt.label = LABEL_POTHOLE_TRENCH;
        points.push_back(pt);
    }
    // Also add some ground-level points nearby for ΔZ to register
    for (int i = 0; i < count / 2; ++i) {
        Point4 pt;
        pt.x = cx + jitter(rng);
        pt.y = cy + jitter(rng);
        pt.z = 0.0f + jitter(rng) * 0.05f;
        pt.label = LABEL_ROAD;
        points.push_back(pt);
    }
}

// ─── Tests ───────────────────────────────────────────────────────────

static int test_compression_ratio() {
    printf("  [TEST] Compression ratio... ");

    auto engine = create_engine();
    auto points = generate_test_points(100000);
    auto result = engine->process(points.data(), points.size());

    printf("N=%d → M=%d (%.1f%% reduction)\n",
           result.raw_point_count, result.compressed_count,
           (1.0 - static_cast<double>(result.compressed_count) / result.raw_point_count) * 100.0);

    if (result.compressed_count >= 100000) {
        printf("  FAIL: No compression achieved\n");
        return 1;
    }
    if (result.compressed_count > 15000) {
        printf("  WARN: M > 15000, but acceptable for random data\n");
    }

    printf("  PASS\n");
    return 0;
}

static int test_output_shape() {
    printf("  [TEST] Output cell structure... ");

    auto engine = create_engine();
    auto points = generate_test_points(1000);
    auto result = engine->process(points.data(), points.size());

    for (const auto& cell : result.cells) {
        // Verify all fields are populated
        if (std::isnan(cell.center_x) || std::isnan(cell.center_y) ||
            std::isnan(cell.max_z) || std::isnan(cell.delta_z) ||
            std::isnan(cell.radius)) {
            printf("  FAIL: NaN detected in cell output\n");
            return 1;
        }
        if (cell.label < 0 || cell.label >= LABEL_COUNT) {
            printf("  FAIL: Invalid label %d\n", cell.label);
            return 1;
        }
        if (cell.delta_z < 0.0f) {
            printf("  FAIL: Negative delta_z %.4f\n", cell.delta_z);
            return 1;
        }
    }

    printf("PASS (%zu cells)\n", result.cells.size());
    return 0;
}

static int test_pothole_detection() {
    printf("  [TEST] Negative obstacle (pothole) detection... ");

    auto engine = create_engine();
    auto points = generate_test_points(10000);

    // Plant a deep pothole at (5, 5) with 0.5m depth — well above threshold
    add_pothole(points, 5.0f, 5.0f, 0.5f, 200);

    auto result = engine->process(points.data(), points.size());

    if (result.threats.empty()) {
        printf("  FAIL: No threats detected\n");
        return 1;
    }

    // Check that at least one threat is near our planted pothole
    bool found_pothole = false;
    for (const auto& t : result.threats) {
        float dx = t.coordinates[0] - 5.0f;
        float dy = t.coordinates[1] - 5.0f;
        float dist = std::sqrt(dx * dx + dy * dy);
        if (dist < 2.0f && t.depth_m >= 0.3f) {
            found_pothole = true;
            printf("found at (%.2f, %.2f) depth=%.2fm ",
                   t.coordinates[0], t.coordinates[1], t.depth_m);
            break;
        }
    }

    if (!found_pothole) {
        printf("  FAIL: Planted pothole not detected in threats\n");
        return 1;
    }

    printf("PASS (%zu threats total)\n", result.threats.size());
    return 0;
}

static int test_label_tie_breaking() {
    printf("  [TEST] Label tie-breaking (hazard priority)... ");

    // Create a small cluster where labels are evenly split
    CellAccumulator acc;
    // Equal counts: 5 Road, 5 Pothole
    for (int i = 0; i < 5; ++i) {
        acc.add_point(1.0f, 1.0f, 0.0f, LABEL_ROAD);
        acc.add_point(1.0f, 1.0f, -0.3f, LABEL_POTHOLE_TRENCH);
    }

    int32_t resolved = acc.resolve_label();
    if (resolved != LABEL_POTHOLE_TRENCH) {
        printf("  FAIL: Expected POTHOLE_TRENCH (%d), got %d\n",
               LABEL_POTHOLE_TRENCH, resolved);
        return 1;
    }

    printf("PASS (resolved to label %d = Pothole/Trench)\n", resolved);
    return 0;
}

static int test_latency() {
    printf("  [TEST] Processing latency (N=100000)... ");

    auto engine = create_engine();
    auto points = generate_test_points(100000);

    // Warm up
    engine->process(points.data(), points.size());

    // Benchmark
    const int RUNS = 5;
    double total_ms = 0.0;
    for (int i = 0; i < RUNS; ++i) {
        auto result = engine->process(points.data(), points.size());
        total_ms += result.latency_ms;
    }
    double avg_ms = total_ms / RUNS;

    printf("%.2f ms avg (%s)\n", avg_ms, engine->info().c_str());

    if (engine->tier() == ComputeTier::NUMBA_TIER_2 && avg_ms > 50.0) {
        printf("  WARN: CPU path > 50ms — target is < 20ms\n");
    }
    if (engine->tier() == ComputeTier::CUDA_TIER_1 && avg_ms > 10.0) {
        printf("  WARN: CUDA path > 10ms — target is < 5ms\n");
    }

    printf("  PASS\n");
    return 0;
}

// ─── Main ────────────────────────────────────────────────────────────

int main() {
    printf("╔══════════════════════════════════════╗\n");
    printf("║  KAVACH-2.5D Engine Tests             ║\n");
    printf("╚══════════════════════════════════════╝\n\n");

    auto engine = create_engine();
    printf("Active engine: %s\n\n", engine->info().c_str());

    int failures = 0;
    failures += test_compression_ratio();
    failures += test_output_shape();
    failures += test_pothole_detection();
    failures += test_label_tie_breaking();
    failures += test_latency();

    printf("\n─────────────────────────────────────────\n");
    if (failures == 0) {
        printf("All tests PASSED ✓\n");
    } else {
        printf("%d test(s) FAILED ✗\n", failures);
    }

    return failures;
}
