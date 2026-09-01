/**
 * KAVACH-2.5D — pybind11 Python Bindings
 *
 * Exposes the C++/CUDA engine to Python as the `kavach_engine` module.
 *
 * Python API:
 *   kavach_engine.process_frame(np.ndarray) -> dict
 *   kavach_engine.get_active_tier() -> str
 *   kavach_engine.get_engine_info() -> str
 */

#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
#include <pybind11/stl.h>

#include "grid_polar.hpp"
#include "config.hpp"

#include <memory>
#include <vector>
#include <string>
#include <stdexcept>

namespace py = pybind11;

namespace {

// Module-level engine singleton (created on first use)
static std::unique_ptr<kavach::GridPolarEngine> g_engine;

kavach::GridPolarEngine& get_engine() {
    if (!g_engine) {
        g_engine = kavach::create_engine();
    }
    return *g_engine;
}

/**
 * Process a frame of LiDAR points.
 *
 * @param points  NumPy array of shape (N, 4) with columns [X, Y, Z, Label_ID].
 *                X, Y, Z are float32; Label_ID is int32 but stored as float32 column.
 * @return dict matching Interface 3 "grid_data" + "threats" + "telemetry" structure.
 */
py::dict process_frame(py::array_t<float, py::array::c_style | py::array::forcecast> points) {
    // Validate input shape
    auto buf = points.request();
    if (buf.ndim != 2 || buf.shape[1] != 4) {
        throw std::invalid_argument(
            "Expected array of shape (N, 4), got (" +
            std::to_string(buf.ndim > 0 ? buf.shape[0] : 0) + ", " +
            std::to_string(buf.ndim > 1 ? buf.shape[1] : 0) + ")"
        );
    }

    size_t n = static_cast<size_t>(buf.shape[0]);
    const float* data = static_cast<const float*>(buf.ptr);

    // Convert from row-major float[N][4] to Point4 array
    // Point4 has the same memory layout as float[3] + int32, but we need to handle
    // the label column (float→int32 cast)
    std::vector<kavach::Point4> point_vec(n);
    for (size_t i = 0; i < n; ++i) {
        point_vec[i].x     = data[i * 4 + 0];
        point_vec[i].y     = data[i * 4 + 1];
        point_vec[i].z     = data[i * 4 + 2];
        point_vec[i].label = static_cast<int32_t>(data[i * 4 + 3]);
    }

    // Process
    kavach::ProcessedFrame frame = get_engine().process(point_vec.data(), n);

    // Build grid_data as list of lists: [[cx, cy, max_z, dz, label, r], ...]
    py::list grid_data;
    for (const auto& cell : frame.cells) {
        py::list row;
        row.append(cell.center_x);
        row.append(cell.center_y);
        row.append(cell.max_z);
        row.append(cell.delta_z);
        row.append(cell.label);
        row.append(cell.radius);
        grid_data.append(row);
    }

    // Build threats list
    py::list threats;
    for (const auto& t : frame.threats) {
        py::dict threat;
        threat["type"] = t.type;
        threat["distance_m"] = t.distance_m;
        py::list coords;
        coords.append(t.coordinates[0]);
        coords.append(t.coordinates[1]);
        threat["coordinates"] = coords;
        threat["depth_m"] = t.depth_m;
        threats.append(threat);
    }

    // Build telemetry
    py::dict telemetry;
    telemetry["latency_ms"] = frame.latency_ms;
    telemetry["raw_points_count"] = frame.raw_point_count;
    telemetry["compressed_cells_count"] = frame.compressed_count;

    // Memory saved: compare raw float32 (N*4*4 bytes) vs compressed (M*6*4 bytes)
    double raw_bytes = static_cast<double>(frame.raw_point_count) * 4.0 * 4.0;
    double compressed_bytes = static_cast<double>(frame.compressed_count) * 6.0 * 4.0;
    double memory_saved = (raw_bytes > 0) ? (1.0 - compressed_bytes / raw_bytes) * 100.0 : 0.0;
    telemetry["memory_saved_percent"] = memory_saved;

    // Build result dict
    py::dict result;
    result["grid_data"] = grid_data;
    result["threats"] = threats;
    result["telemetry"] = telemetry;

    return result;
}

/**
 * Process frame and return grid_data as a NumPy array (M, 6) for zero-copy downstream.
 */
py::array_t<float> process_frame_numpy(
    py::array_t<float, py::array::c_style | py::array::forcecast> points
) {
    auto buf = points.request();
    if (buf.ndim != 2 || buf.shape[1] != 4) {
        throw std::invalid_argument("Expected array of shape (N, 4)");
    }

    size_t n = static_cast<size_t>(buf.shape[0]);
    const float* data = static_cast<const float*>(buf.ptr);

    std::vector<kavach::Point4> point_vec(n);
    for (size_t i = 0; i < n; ++i) {
        point_vec[i].x     = data[i * 4 + 0];
        point_vec[i].y     = data[i * 4 + 1];
        point_vec[i].z     = data[i * 4 + 2];
        point_vec[i].label = static_cast<int32_t>(data[i * 4 + 3]);
    }

    kavach::ProcessedFrame frame = get_engine().process(point_vec.data(), n);

    // Create output NumPy array (M, 6)
    size_t m = frame.cells.size();
    auto result = py::array_t<float>({static_cast<py::ssize_t>(m), py::ssize_t(6)});
    auto r = result.mutable_unchecked<2>();

    for (size_t i = 0; i < m; ++i) {
        const auto& c = frame.cells[i];
        r(i, 0) = c.center_x;
        r(i, 1) = c.center_y;
        r(i, 2) = c.max_z;
        r(i, 3) = c.delta_z;
        r(i, 4) = static_cast<float>(c.label);
        r(i, 5) = c.radius;
    }

    return result;
}

std::string get_active_tier() {
    return kavach::tier_to_string(get_engine().tier());
}

std::string get_engine_info() {
    return get_engine().info();
}

} // anonymous namespace

// ─── Module Definition ───────────────────────────────────────────────

PYBIND11_MODULE(kavach_engine, m) {
    m.doc() = "KAVACH-2.5D Foveated Elevation & Semantic Mapping Engine";

    m.def("process_frame", &process_frame,
          py::arg("points"),
          R"doc(
          Process a LiDAR point cloud frame.

          Args:
              points: NumPy array of shape (N, 4) — [X, Y, Z, Label_ID] per row.
                      X, Y, Z in meters (float32), Label_ID as int (0-3).

          Returns:
              dict with keys:
                  "grid_data": list of [cx, cy, max_z, delta_z, label, radius] per cell
                  "threats": list of threat dicts
                  "telemetry": dict with latency_ms, raw_points_count, compressed_cells_count, memory_saved_percent
          )doc"
    );

    m.def("process_frame_numpy", &process_frame_numpy,
          py::arg("points"),
          "Process frame and return grid_data as NumPy array (M, 6) for zero-copy use."
    );

    m.def("get_active_tier", &get_active_tier,
          "Get the currently active compute tier: 'CUDA_TIER_1' or 'NUMBA_TIER_2'."
    );

    m.def("get_engine_info", &get_engine_info,
          "Get human-readable description of the active engine."
    );

    // Expose constants
    m.attr("DELTA_Z_THRESHOLD") = kavach::DELTA_Z_THRESHOLD;
    m.attr("NEAR_FIELD_MAX") = kavach::NEAR_FIELD_MAX;
    m.attr("MID_FIELD_MAX") = kavach::MID_FIELD_MAX;
    m.attr("FAR_FIELD_MAX") = kavach::FAR_FIELD_MAX;
}
