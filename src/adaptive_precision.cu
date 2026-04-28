/**
 * Adaptive Precision Controller + Constraint-Aware Scheduler + Batch Constraint Propagation
 *
 * Targets: Jetson Orin Nano (SM 8.7) + RTX 4050 (SM 8.9)
 * Novel aspects:
 *   1. Runtime precision switching based on thermal, power, accuracy, and memory pressure
 *   2. GPU-parallel multi-objective task-node scoring with hard constraint feasibility
 *   3. Warp-level shuffle reduction for batch navigation constraint propagation
 *
 * Author: Forgemaster ⚒️ — Cocapn fleet
 * License: MIT
 */

#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstdint>
#include <cstring>

#include "marine_types.h"

// ------------------------------------------------------------------
// Local error-checking macro for functions returning int (not cudaError_t)
// ------------------------------------------------------------------
#define CUDA_CHECK_RET_INT(call)                                                \
    do {                                                                        \
        cudaError_t _err = (call);                                              \
        if (_err != cudaSuccess) {                                              \
            fprintf(stderr, "CUDA error at %s:%d — %s: %s\n",                   \
                    __FILE__, __LINE__, #call, cudaGetErrorString(_err));       \
            return -1;                                                          \
        }                                                                       \
    } while (0)

// ------------------------------------------------------------------
// 1. RUNTIME PRECISION CONTROLLER
// ------------------------------------------------------------------

/**
 * @brief Select compute precision at runtime based on environmental and accuracy constraints.
 *
 * Novel: Hierarchical decision — hard constraints (thermal/power) override accuracy
 * preferences. TF32 is selected as a middle ground when accuracy is marginal but
 * thermal headroom exists, exploiting Ada/Ampere tensor cores without FP16
 * numerical drift.
 */
__device__ __host__ ComputePrecision select_precision_runtime(
    float gpu_temp_c,
    float power_usage_w,
    float power_budget_w,
    float position_error_m,
    float memory_pressure,
    const AdaptivePrecisionConfig* config
)
{
    if (!config) {
        return PREC_FP32;
    }

    // Hard safety constraints: if we are at or above thermal limit, or already
    // at power budget, drop to FP16 immediately to reduce register pressure
    // and memory bandwidth (roughly halves both vs FP32).
    if (config->thermal_threshold > 0.0f && gpu_temp_c >= config->thermal_threshold) {
        return PREC_FP16;
    }
    if (config->power_budget > 0.0f && power_usage_w >= config->power_budget) {
        return PREC_FP16;
    }

    // Memory pressure: high pressure means less bandwidth and potential paging.
    // FP16 halves the footprint and bandwidth.
    if (config->memory_pressure_threshold > 0.0f &&
        memory_pressure > config->memory_pressure_threshold) {
        return PREC_FP16;
    }

    // Accuracy-driven tiered selection
    float target = (config->accuracy_target > 0.0f) ? config->accuracy_target : 1.0f;

    if (position_error_m < target * 0.25f) {
        // Very accurate — comfortable margin for FP16
        return PREC_FP16;
    }
    if (config->tf32_accuracy_margin > 0.0f &&
        position_error_m < target * config->tf32_accuracy_margin) {
        // Marginal accuracy — TF32 gives ~FP32-range with FP16 throughput on SM>=80
        return PREC_TF32;
    }

    // Accuracy is poor or unknown — default to full FP32
    return PREC_FP32;
}

// ------------------------------------------------------------------
// NVML-based Host-side Metrics (optional — compile with -DHAS_NVML)
// ------------------------------------------------------------------
#ifdef HAS_NVML
#include <nvml.h>

static bool g_nvml_initialized = false;

static bool ensure_nvml(void) {
    if (g_nvml_initialized) return true;
    nvmlReturn_t r = nvmlInit();
    if (r == NVML_SUCCESS) {
        g_nvml_initialized = true;
        return true;
    }
    return false;
}

/**
 * @brief Read live GPU metrics via NVML.
 * @return true if NVML succeeded and all outputs are populated.
 */
bool read_gpu_metrics_nvml(int device_id, float* temp, float* power,
                           size_t* mem_free, size_t* mem_total)
{
    if (!ensure_nvml()) return false;

    nvmlDevice_t dev;
    nvmlReturn_t r = nvmlDeviceGetHandleByIndex(device_id, &dev);
    if (r != NVML_SUCCESS) return false;

    unsigned int t = 0;
    r = nvmlDeviceGetTemperature(dev, NVML_TEMPERATURE_GPU, &t);
    if (r != NVML_SUCCESS) return false;
    *temp = static_cast<float>(t);

    unsigned int p = 0;
    r = nvmlDeviceGetPowerUsage(dev, &p);
    if (r != NVML_SUCCESS) return false;
    *power = static_cast<float>(p) * 0.001f; // mW -> W

    nvmlMemory_t mem;
    r = nvmlDeviceGetMemoryInfo(dev, &mem);
    if (r != NVML_SUCCESS) return false;
    *mem_free = mem.free;
    *mem_total = mem.total;

    return true;
}

#else // !HAS_NVML

bool read_gpu_metrics_nvml(int, float*, float*, size_t*, size_t*) {
    return false;
}

#endif // HAS_NVML

// ------------------------------------------------------------------
// 2. CONSTRAINT-AWARE TASK SCHEDULER
// ------------------------------------------------------------------

/**
 * @brief GPU kernel scoring every (task, node) pair in parallel.
 *
 * Novel: Each thread evaluates the full multi-objective score for one pair,
 * including dynamic constraint feasibility (thermal, memory, power, deadline,
 * precision support). The score is a weighted Pareto aggregate so greedy
 * assignment on the host can pick the best feasible node per task.
 *
 * Constraints enforced as hard feasibility filters; score only computed for
 * feasible pairs to avoid silent constraint violations.
 */
__global__ void score_task_node_pairs(
    const ComputeTask* __restrict__ tasks,
    const NodeProfile* __restrict__ nodes,
    TaskNodeScore* __restrict__ scores,
    int num_tasks,
    int num_nodes
)
{
    int node_idx = blockIdx.x * blockDim.x + threadIdx.x;
    int task_idx = blockIdx.y;

    if (node_idx >= num_nodes || task_idx >= num_tasks) return;

    const ComputeTask& task = tasks[task_idx];
    const NodeProfile& node = nodes[node_idx];

    TaskNodeScore result;
    result.score = -1.0f;
    result.feasible = 0;
    result.predicted_latency_ms = 0.0f;
    result.predicted_power_w = 0.0f;

    // --- Hard constraint checks ---
    float thermal_headroom = node.temp_limit_c - node.temp_c;
    if (thermal_headroom <= 0.0f) {
        scores[task_idx * num_nodes + node_idx] = result;
        return;
    }

    size_t memory_free = (node.memory_total > node.memory_used)
                         ? (node.memory_total - node.memory_used) : 0;
    if (memory_free < task.memory_bytes) {
        scores[task_idx * num_nodes + node_idx] = result;
        return;
    }

    bool precision_ok = (node.precision_mask &
                         (1u << static_cast<unsigned int>(task.precision_req))) != 0;
    if (!precision_ok) {
        scores[task_idx * num_nodes + node_idx] = result;
        return;
    }

    // --- Predicted latency (compute + network) ---
    // task.compute_gflops = GFLOPs of work, node.peak_gflops = TFLOPS
    // compute_time_s = work / (rate * 1e3) then convert to ms
    float compute_time_ms = (task.compute_gflops / fmaxf(node.peak_gflops * 1000.0f, 0.001f)) * 1000.0f;
    float predicted_latency = compute_time_ms + node.network_latency_ms;

    if (predicted_latency >= task.deadline_ms) {
        scores[task_idx * num_nodes + node_idx] = result;
        return;
    }

    // --- Predicted power (affine model: base + utilization * headroom) ---
    // utilization = fraction of peak TFLOPS this task demands
    float utilization = task.compute_gflops / fmaxf(node.peak_gflops * 1000.0f, 0.001f);
    float predicted_power = node.power_w + utilization *
                            (node.power_limit_w - node.power_w) * 0.6f;

    if (predicted_power > node.power_limit_w) {
        scores[task_idx * num_nodes + node_idx] = result;
        return;
    }

    result.feasible = 1;
    result.predicted_latency_ms = predicted_latency;
    result.predicted_power_w = predicted_power;

    // --- Multi-objective score ---
    // Normalized sub-objectives in [0, 1], higher is better
    float perf_score    = fminf(task.compute_gflops /
                                fmaxf(node.peak_gflops, 1.0f), 1.0f);
    float thermal_score = fmaxf(thermal_headroom / 30.0f, 0.0f);
    float latency_score = fmaxf((task.deadline_ms - predicted_latency) /
                                fmaxf(task.deadline_ms, 1.0f), 0.0f);
    float power_score   = fmaxf((node.power_limit_w - predicted_power) /
                                fmaxf(node.power_limit_w, 1.0f), 0.0f);
    float accuracy_score = fmaxf(task.accuracy_target /
                                 fmaxf(task.accuracy_target + 0.1f, 0.1f), 0.0f);

    const float w_perf = 0.20f;
    const float w_thermal = 0.25f;
    const float w_latency = 0.30f;
    const float w_power = 0.15f;
    const float w_accuracy = 0.10f;

    result.score = w_perf * perf_score
                 + w_thermal * thermal_score
                 + w_latency * latency_score
                 + w_power * power_score
                 + w_accuracy * accuracy_score;

    // Prefer Jetson for FP16 inference (better perf-per-watt tensor core utilization)
    if (node.is_jetson && task.precision_req == PREC_FP16) {
        result.score *= 1.15f;
    }

    scores[task_idx * num_nodes + node_idx] = result;
}

/**
 * @brief Host-side greedy assignment using GPU-computed scores.
 *
 * Takes host-visible task and node descriptions, copies them to the device,
 * launches the scoring kernel, then greedily assigns each task to the best
 * feasible node accounting for accumulated power load.
 *
 * @return 0 on success, -1 on error.
 */
extern "C" __host__ int schedule_tasks_greedy(
    const ComputeTask* h_tasks,
    const NodeProfile* h_nodes,
    int num_tasks,
    int num_nodes,
    int* assignment,       // [num_tasks] output: chosen node index, -1 if unassigned
    cudaStream_t stream
)
{
    if (!h_tasks || !h_nodes || !assignment || num_tasks <= 0 || num_nodes <= 0) {
        return -1;
    }

    memset(assignment, -1, num_tasks * sizeof(int));

    ComputeTask* d_tasks = nullptr;
    NodeProfile* d_nodes = nullptr;
    TaskNodeScore* d_scores = nullptr;
    TaskNodeScore* h_scores = nullptr;
    float* acc_power = nullptr;
    size_t* acc_mem = nullptr;
    cudaError_t err = cudaSuccess;

    auto do_cleanup = [&]() {
        free(h_scores);
        free(acc_power);
        free(acc_mem);
        if (d_tasks) cudaFreeAsync(d_tasks, stream);
        if (d_nodes) cudaFreeAsync(d_nodes, stream);
        if (d_scores) cudaFreeAsync(d_scores, stream);
    };

    err = cudaMallocAsync(&d_tasks, num_tasks * sizeof(ComputeTask), stream);
    if (err != cudaSuccess) { do_cleanup(); return -1; }

    err = cudaMallocAsync(&d_nodes, num_nodes * sizeof(NodeProfile), stream);
    if (err != cudaSuccess) { do_cleanup(); return -1; }

    err = cudaMallocAsync(&d_scores, num_tasks * num_nodes * sizeof(TaskNodeScore), stream);
    if (err != cudaSuccess) { do_cleanup(); return -1; }

    err = cudaMemcpyAsync(d_tasks, h_tasks, num_tasks * sizeof(ComputeTask),
                          cudaMemcpyHostToDevice, stream);
    if (err != cudaSuccess) { do_cleanup(); return -1; }

    err = cudaMemcpyAsync(d_nodes, h_nodes, num_nodes * sizeof(NodeProfile),
                          cudaMemcpyHostToDevice, stream);
    if (err != cudaSuccess) { do_cleanup(); return -1; }

    dim3 threads(128, 1);
    dim3 blocks((num_nodes + threads.x - 1) / threads.x, num_tasks);
    score_task_node_pairs<<<blocks, threads, 0, stream>>>(
        d_tasks, d_nodes, d_scores, num_tasks, num_nodes
    );

    err = cudaGetLastError();
    if (err != cudaSuccess) { do_cleanup(); return -1; }

    h_scores = (TaskNodeScore*)malloc(num_tasks * num_nodes * sizeof(TaskNodeScore));
    if (!h_scores) { do_cleanup(); return -1; }

    err = cudaMemcpyAsync(h_scores, d_scores,
                          num_tasks * num_nodes * sizeof(TaskNodeScore),
                          cudaMemcpyDeviceToHost, stream);
    if (err != cudaSuccess) { do_cleanup(); return -1; }

    err = cudaStreamSynchronize(stream);
    if (err != cudaSuccess) { do_cleanup(); return -1; }

    acc_power = (float*)calloc(num_nodes, sizeof(float));
    acc_mem   = (size_t*)calloc(num_nodes, sizeof(size_t));
    if (!acc_power || !acc_mem) { do_cleanup(); return -1; }

    for (int t = 0; t < num_tasks; ++t) {
        int best_node = -1;
        float best_score = -1.0f;

        for (int n = 0; n < num_nodes; ++n) {
            const TaskNodeScore& s = h_scores[t * num_nodes + n];
            if (!s.feasible || s.score <= best_score) continue;

            // Dynamic load check with accumulated state
            // acc_power tracks incremental power per task, not absolute
            float power_delta = s.predicted_power_w - h_nodes[n].power_w;
            if (acc_power[n] + power_delta > h_nodes[n].power_limit_w - h_nodes[n].power_w) continue;
            size_t mem_avail = (h_nodes[n].memory_total > h_nodes[n].memory_used)
                               ? (h_nodes[n].memory_total - h_nodes[n].memory_used) : 0;
            if (acc_mem[n] + h_tasks[t].memory_bytes > mem_avail) continue;

            best_score = s.score;
            best_node = n;
        }

        assignment[t] = best_node;
        if (best_node >= 0) {
            float power_delta = h_scores[t * num_nodes + best_node].predicted_power_w - h_nodes[best_node].power_w;
            acc_power[best_node] += power_delta;
            acc_mem[best_node]   += h_tasks[t].memory_bytes;
        }
    }

    do_cleanup();
    return 0;
}

// ------------------------------------------------------------------
// 3. BATCH CONSTRAINT PROPAGATION KERNEL
// ------------------------------------------------------------------

/**
 * @brief Evaluate a single navigation constraint against a state.
 *
 * Returns the violation amount, severity, and constraint ID.
 */
__device__ __forceinline__ ConstraintResult evaluate_single_constraint(
    const NavState& state,
    const NavConstraint& constraint,
    uint8_t constraint_id
)
{
    ConstraintResult r;
    r.violation_amount = 0.0f;
    r.constraint_id = constraint_id;
    r.violated = 0;
    r.severity = 0;
    r.pad = 0;

    if (!constraint.active) {
        return r;
    }

    // Speed over ground in knots
    float vn = __half2float(state.vel.x);
    float ve = __half2float(state.vel.y);
    float speed_kts = sqrtf(vn * vn + ve * ve) * 1.94384f;

    // 1. Depth constraint
    if (constraint.min_depth > 0.0f) {
        float v = constraint.min_depth - state.depth;
        if (v > r.violation_amount) {
            r.violation_amount = v;
            r.violated = 1;
            r.severity = (v > 2.0f) ? 2 : 1;
        }
    }

    // 2. Speed constraint
    if (constraint.max_speed > 0.0f && speed_kts > constraint.max_speed) {
        float v = speed_kts - constraint.max_speed;
        if (v > r.violation_amount) {
            r.violation_amount = v;
            r.violated = 1;
            r.severity = (v > 5.0f) ? 2 : 1;
        }
    }

    // 3. Safety radius (hazard at origin for simplified check;
    //    real systems pass hazard list via extended constraint struct)
    if (constraint.safety_radius > 0.0f) {
        float lat = __half2float(state.pos.x);
        float lon = __half2float(state.pos.y);
        const float R = 6371000.0f;
        float lat_rad = lat * 3.14159265f / 180.0f;
        float x = lon * 3.14159265f / 180.0f * cosf(lat_rad) * R;
        float y = lat_rad * R;
        float dist = sqrtf(x * x + y * y);

        if (dist < constraint.safety_radius) {
            float v = constraint.safety_radius - dist;
            if (v > r.violation_amount) {
                r.violation_amount = v;
                r.violated = 1;
                r.severity = 2; // safety radius violation is always critical
            }
        }
    }

    return r;
}

/**
 * @brief Batch constraint propagation with warp-level reduction.
 *
 * Novel: One block per NavState. Threads stride over constraints, then each
 * warp performs a shuffle reduction to find its worst violation without
 * atomic contention. Per-warp aggregates are reduced in shared memory by
 * thread 0. This minimizes shared-memory bank conflicts and avoids global
 * atomics entirely.
 *
 * Each thread handles one constraint (or more via striding) for one state.
 */
__global__ void batch_constraint_propagation(
    const NavState* __restrict__ states,
    const NavConstraint* __restrict__ constraints,
    ConstraintResult* __restrict__ results,
    int num_constraints,
    int num_states
)
{
    int state_idx = blockIdx.x;
    if (state_idx >= num_states) return;

    int tid = threadIdx.x;
    int lane_id = tid & 31;
    int warp_id = tid >> 5;
    int num_warps = blockDim.x >> 5;

    // Dynamic shared memory: float per warp for violation, uint8 for severity & cid
    extern __shared__ float s_mem[];
    float* s_viol = s_mem;                         // [num_warps]
    uint8_t* s_sev = (uint8_t*)&s_viol[num_warps]; // [num_warps]
    uint8_t* s_cid = (uint8_t*)&s_sev[num_warps];  // [num_warps]

    if (tid < num_warps) {
        s_viol[tid] = 0.0f;
        s_sev[tid] = 0;
        s_cid[tid] = 0;
    }
    __syncthreads();

    const NavState state = states[state_idx];

    // Strided constraint evaluation
    float local_viol = 0.0f;
    uint8_t local_sev = 0;
    uint8_t local_cid = 0;

    for (int c = tid; c < num_constraints; c += blockDim.x) {
        ConstraintResult r = evaluate_single_constraint(
            state, constraints[c], static_cast<uint8_t>(c)
        );
        if (r.violation_amount > local_viol) {
            local_viol = r.violation_amount;
            local_sev = r.severity;
            local_cid = r.constraint_id;
        }
    }

    // Pack severity and constraint ID into a single unsigned int to halve shuffle ops
    unsigned int packed = (static_cast<unsigned int>(local_sev) << 16)
                        | static_cast<unsigned int>(local_cid);

    // Warp reduction via shuffle — no shared memory atomics, O(log warpSize) steps
    #pragma unroll
    for (int offset = 16; offset > 0; offset >>= 1) {
        float other_viol = __shfl_down_sync(0xFFFFFFFF, local_viol, offset);
        unsigned int other_packed = __shfl_down_sync(0xFFFFFFFF, packed, offset);

        if (other_viol > local_viol) {
            local_viol = other_viol;
            packed = other_packed;
        }
    }

    local_sev = static_cast<uint8_t>((packed >> 16) & 0xFF);
    local_cid = static_cast<uint8_t>(packed & 0xFF);

    // Lane 0 commits warp aggregate to shared memory
    if (lane_id == 0) {
        s_viol[warp_id] = local_viol;
        s_sev[warp_id] = local_sev;
        s_cid[warp_id] = local_cid;
    }
    __syncthreads();

    // Thread 0 reduces across warps and writes the final worst violation
    if (tid == 0) {
        float worst_viol = 0.0f;
        uint8_t worst_sev = 0;
        uint8_t worst_cid = 0;

        for (int w = 0; w < num_warps; ++w) {
            if (s_viol[w] > worst_viol) {
                worst_viol = s_viol[w];
                worst_sev = s_sev[w];
                worst_cid = s_cid[w];
            }
        }

        ConstraintResult out;
        out.violation_amount = worst_viol;
        out.violated = (worst_viol > 0.0f) ? 1 : 0;
        out.severity = worst_sev;
        out.constraint_id = worst_cid;
        out.pad = 0;

        results[state_idx] = out;
    }
}

// ------------------------------------------------------------------
// Host Wrappers
// ------------------------------------------------------------------
extern "C" {

/**
 * @brief Host wrapper for batch constraint propagation.
 *
 * Launches one block per state. Shared memory sized for per-warp aggregates.
 */
cudaError_t marine_batch_constraints(
    const NavState* d_states,
    const NavConstraint* d_constraints,
    ConstraintResult* d_results,
    int num_constraints,
    int num_states,
    cudaStream_t stream
)
{
    if (!d_states || !d_constraints || !d_results ||
        num_states <= 0 || num_constraints <= 0) {
        return cudaErrorInvalidValue;
    }

    int block_size = 128; // 4 warps
    int warps = block_size / 32;
    size_t shared_size = warps * sizeof(float) + warps * 2 * sizeof(uint8_t);

    batch_constraint_propagation<<<num_states, block_size, shared_size, stream>>>(
        d_states, d_constraints, d_results, num_constraints, num_states
    );

    CUDA_CHECK(cudaGetLastError());
    return cudaSuccess;
}

} // extern "C"
