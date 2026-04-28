/**
 * marine_types.h — Shared types for marine GPU edge computing
 *
 * All structs used across CUDA kernels, MEP bridge, and benchmarks.
 * Safe to include from both CUDA (.cu) and C++ (.cpp) translation units.
 *
 * Author: Forgemaster ⚒️
 * License: MIT
 */

#pragma once

#include <cstddef>
#include <cstdint>

// cuda_fp16.h provides __half / half2 — only available when compiling with nvcc.
// Plain C++ consumers (mep_bridge.cpp) must not dereference FP16 fields.
#ifdef __CUDACC__
#  include <cuda_fp16.h>
#endif

// ============================================================
// HARDWARE CONFIG — Adaptive for target hardware
// ============================================================

#ifdef JETSON_ORIN
#  define MEP_MAX_SENSORS    16
#  define MEP_MAX_WATERFALL  4096
#else
#  define MEP_MAX_SENSORS    64
#  define MEP_MAX_WATERFALL  16384
#endif

// ============================================================
// CUDA ERROR CHECKING
// ============================================================

#ifdef __CUDACC__
#  include <cuda_runtime.h>
#  include <cstdio>

#  define CUDA_CHECK(call)                                                      \
    do {                                                                        \
        cudaError_t _err = (call);                                              \
        if (_err != cudaSuccess) {                                              \
            fprintf(stderr, "CUDA error at %s:%d — %s: %s\n",                 \
                    __FILE__, __LINE__, #call, cudaGetErrorString(_err));       \
            return _err;                                                        \
        }                                                                       \
    } while (0)

#  define CUDA_CHECK_VOID(call)                                                 \
    do {                                                                        \
        cudaError_t _err = (call);                                              \
        if (_err != cudaSuccess) {                                              \
            fprintf(stderr, "CUDA error at %s:%d — %s: %s\n",                 \
                    __FILE__, __LINE__, #call, cudaGetErrorString(_err));       \
            return;                                                             \
        }                                                                       \
    } while (0)
#endif // __CUDACC__

// ============================================================
// NMEA PARSING
// ============================================================

struct NMEAParsed {
    float    timestamp;       // seconds since epoch
    float    lat;             // decimal degrees
    float    lon;             // decimal degrees
    float    sog;             // speed over ground (knots)
    float    cog;             // course over ground (degrees)
    uint8_t  sentence_type;   // 0=unknown 1=GGA 2=RMC 3=HDT 4=VHW 5=DBT
    uint8_t  valid;           // 1 if checksum OK + data complete
    uint8_t  pad[2];
};

// Fixed-width per-sentence buffer for the GPU batch kernel.
// All sentences are padded to this stride so the kernel can compute
// offsets as  sentences + idx * NMEA_SENTENCE_STRIDE  without a separate
// offset table (avoids an extra indirection per thread).
#define NMEA_SENTENCE_STRIDE 128   // bytes — covers longest NMEA sentence

// ============================================================
// NAVIGATION STATE
// ============================================================

#ifdef __CUDACC__
struct NavState {
    half2   pos;           // FP16: lat, lon (degrees)
    half2   vel;           // FP16: north vel, east vel (m/s)
    half    heading;       // FP16: heading (radians)
    half    heading_rate;  // FP16: heading rate (rad/s)
    float   depth;         // FP32: sonar depth (metres) — precision-critical
    float   timestamp;     // seconds
};
#else
// Plain C++ layout — FP16 fields stored as uint16_t pairs; not for arithmetic.
struct NavState {
    uint16_t pos[2];
    uint16_t vel[2];
    uint16_t heading;
    uint16_t heading_rate;
    float    depth;
    float    timestamp;
};
#endif

// Upper triangle of a 6×6 symmetric matrix stored in FP16 (21 elements).
// Compact covariance representation optimised for Jetson VRAM bandwidth.
struct CovarianceFP16 {
#ifdef __CUDACC__
    half data[21];
#else
    uint16_t data[21];
#endif
    uint8_t pad[2]; // align to 4 bytes
};

// ============================================================
// NAVIGATION CONSTRAINTS
// ============================================================

struct NavConstraint {
    float   min_depth;      // metres — vessel must stay above this
    float   max_speed;      // knots
    float   safety_radius;  // metres from charted hazards
    float   max_roll;       // degrees
    float   min_battery;    // percentage
    uint8_t active;
    uint8_t pad[3];
};

struct ConstraintResult {
    float   violation_amount;  // 0 = satisfied, >0 = magnitude of violation
    uint8_t constraint_id;
    uint8_t violated;
    uint8_t severity;          // 0=ok  1=warning  2=critical
    uint8_t pad;
};

// ============================================================
// ADAPTIVE PRECISION
// ============================================================

enum ComputePrecision : uint8_t {
    PREC_FP32 = 0,
    PREC_FP16 = 1,
    PREC_TF32 = 2,
    PREC_BF16 = 3
};

struct AdaptivePrecisionConfig {
    ComputePrecision kalman_precision;
    ComputePrecision sonar_precision;
    ComputePrecision constraint_precision;
    uint8_t          pad;
    float            thermal_threshold;        // °C — drop precision above this
    float            power_budget;             // watts
    float            accuracy_target;          // max acceptable position error (m)
    float            memory_pressure_threshold;// 0.0–1.0, drop precision when exceeded
    float            tf32_accuracy_margin;     // fraction of accuracy_target where TF32 is acceptable
};

// ============================================================
// SCHEDULER TYPES — Constraint-aware task placement
// ============================================================

struct ComputeTask {
    float            compute_gflops;
    size_t           memory_bytes;
    float            deadline_ms;
    ComputePrecision precision_req;
    uint8_t          pad[3];
    float            accuracy_target;
    float            priority;
};

struct NodeProfile {
    int      node_id;
    float    peak_gflops;
    size_t   memory_total;
    size_t   memory_used;
    float    temp_c;
    float    temp_limit_c;
    float    power_w;
    float    power_limit_w;
    float    network_latency_ms;
    uint32_t precision_mask; // bitmask of supported ComputePrecision values
    uint8_t  is_jetson;
    uint8_t  pad[3];
};

struct TaskNodeScore {
    float    score;
    uint8_t  feasible; // 1 = feasible, 0 = infeasible
    uint8_t  pad[3];
    float    predicted_latency_ms;
    float    predicted_power_w;
};

// ============================================================
// MARINE EDGE PROTOCOL (MEP) — 12-byte binary header
// ============================================================

#define MEP_PORT    7847
#define MEP_MAGIC   0x4D524E45u   // "MRNE"
#define MEP_VERSION 1

enum MEPType : uint16_t {
    // Discovery
    MEP_PING             = 0x0001,
    MEP_PONG             = 0x0002,
    MEP_DEVICE_INFO      = 0x0003,

    // Task distribution
    MEP_TASK_SUBMIT      = 0x0010,
    MEP_TASK_RESULT      = 0x0011,
    MEP_TASK_CANCEL      = 0x0012,
    MEP_TASK_HEARTBEAT   = 0x0013,

    // GPU-specific
    MEP_GPU_OFFLOAD      = 0x0020,
    MEP_GPU_RESULT       = 0x0021,
    MEP_GPU_PTX          = 0x0022,
    MEP_GPU_MEM_XFER     = 0x0023,

    // Marine sensor data
    MEP_SENSOR_BATCH     = 0x0030,
    MEP_NAV_STATE        = 0x0031,
    MEP_SONAR_FRAME      = 0x0032,
    MEP_ALERT            = 0x0033,

    // Constraint-aware scheduling
    MEP_CONSTRAINT_UPDATE  = 0x0040,
    MEP_LOAD_REPORT        = 0x0041,
    MEP_SCHEDULE_REQUEST   = 0x0042,
    MEP_SCHEDULE_ASSIGN    = 0x0043,
};

struct __attribute__((packed)) MEPHeader {
    uint32_t magic;    // MEP_MAGIC
    uint16_t version;  // MEP_VERSION
    uint16_t type;     // MEPType
    uint32_t length;   // payload bytes (0 for header-only messages)
    uint32_t seq;      // sequence number for reorder detection
};

// Static assertion — header must be exactly 16 bytes.
// (Packed: 4+2+2+4+4 = 16.)
static_assert(sizeof(MEPHeader) == 16, "MEPHeader must be 16 bytes");

struct MEPDeviceInfo {
    char     hostname[32];
    char     gpu_name[64];
    uint32_t sm_version;    // e.g. 0x0870 (Orin SM 8.7), 0x0890 (Ada SM 8.9)
    uint64_t total_memory;  // GPU memory bytes
    uint64_t free_memory;
    float    temperature;   // GPU °C
    float    power_usage;   // watts
    float    power_limit;   // watts
    uint32_t max_threads;   // max threads per block
    uint32_t clock_rate;    // kHz
    uint32_t num_sms;       // streaming multiprocessor count
    uint8_t  is_jetson;     // 1 = Jetson (thermal-limited), 0 = desktop
    uint8_t  compute_mode;
    uint8_t  pad[2];
};

struct MEPTaskSubmit {
    uint32_t task_id;
    uint32_t kernel_id;
    uint32_t grid_x,  grid_y,  grid_z;
    uint32_t block_x, block_y, block_z;
    uint32_t shared_mem;    // bytes of dynamic shared memory
    uint64_t input_addr;    // remote GPU address (if pre-staged)
    uint64_t output_addr;
    uint32_t input_size;    // bytes to transfer
    uint32_t output_size;
    float    priority;      // 0.0 (low) – 1.0 (high)
    float    deadline;      // seconds until deadline (0 = no deadline)
    uint8_t  precision;     // ComputePrecision
    uint8_t  flags;         // bit 0: async  bit 1: pipeline  bit 2: PTX kernel
    uint8_t  pad[2];
};

// ============================================================
// FUSION PIPELINE DESCRIPTOR
// ============================================================

enum FusionStage : uint8_t {
    STAGE_NMEA_PARSE       = 0,
    STAGE_TIMESTAMP_ALIGN  = 1,
    STAGE_KALMAN_PREDICT   = 2,
    STAGE_KALMAN_UPDATE    = 3,
    STAGE_SONAR_PROCESS    = 4,
    STAGE_CONSTRAINT_CHECK = 5,
    STAGE_OUTPUT           = 6,
};

struct FusionPipeline {
    FusionStage stages[8];
    int         num_stages;
    int         split_point;
};

// ============================================================
// HOST WRAPPER DECLARATIONS — defined in marine_sensor_fusion.cu
// ============================================================
// These are extern "C" functions callable from both CUDA and C++ code.
// The implementations live in marine_sensor_fusion.cu.

// Pipeline config/result structs (defined in fusion_pipeline.cu, used by benchmarks)
struct FusionConfig {
    bool enable_nmea_parse    = true;
    bool enable_kalman        = true;
    bool enable_sonar         = true;
    bool enable_constraints   = true;
    int  num_sentences        = 10000;
    float base_timestamp      = 0.0f;
    float dt                  = 0.1f;
    int  kalman_steps         = 1;
    int  samples_per_ping     = 2048;
    int  waterfall_width      = 512;
    float time_per_sample     = 0.00005f;
    float sound_velocity      = 1500.0f;
    int  num_constraints      = 8;
};

struct FusionResult {
    float time_nmea    = 0.0f;
    float time_kalman  = 0.0f;
    float time_sonar   = 0.0f;
    float time_constr  = 0.0f;
    float time_total   = 0.0f;
    float nmea_per_sec      = 0.0f;
    float kalman_per_sec    = 0.0f;
    float pings_per_sec     = 0.0f;
    int   violations_found  = 0;
    int   critical_violations = 0;
    float graph_overhead_ms = 0.0f;
    bool  graph_captured    = false;
};
// ============================================================
struct BeamformerConfig {
    int num_elements     = 32;
    int num_beams        = 128;
    int num_samples      = 2048;
    float element_spacing = 0.02f;   // meters
    float sound_velocity = 1500.0f;  // m/s
    float sample_rate    = 40000.0f; // Hz
    float beam_width_deg = 180.0f;
};

// HOST KERNEL WRAPPER DECLARATIONS
// Only visible when compiled by nvcc (cudaError_t / cudaStream_t require
// cuda_runtime.h which non-CUDA translation units do not include).
// Implementations live in marine_sensor_fusion.cu (marine_fusion library).
// ============================================================
#ifdef __CUDACC__
extern "C" {

cudaError_t marine_parse_nmea(
    const char* sentences, const int* lengths, int num_sentences,
    NMEAParsed* output, float base_timestamp, cudaStream_t stream);

cudaError_t marine_kalman_predict(
    NavState* states, CovarianceFP16* covariances, float dt, int count,
    cudaStream_t stream);

cudaError_t marine_sonar_waterfall(
    const float* raw_returns, float* waterfall,
    int samples_per_ping, int waterfall_width,
    float time_per_sample, float sound_velocity, int num_pings,
    cudaStream_t stream);

cudaError_t marine_check_constraints(
    const NavState* states, const NavConstraint* constraints,
    ConstraintResult* results, int num_constraints, int num_states,
    cudaStream_t stream);

} // extern "C"
#endif // __CUDACC__

// ============================================================
// FUSION PIPELINE I/O — used by fusion_pipeline.cu and its benchmarks
// ============================================================

// All device pointers below must be allocated with cudaMalloc before calling
// run_fusion_pipeline(). Output pointers (d_parsed, d_waterfall, d_results)
// must also be allocated by the caller.
struct FusionInput {
    // Stage 0: NMEA parse
    const char*       d_sentences;    // device: [num_sentences * NMEA_SENTENCE_STRIDE]
    const int*        d_lengths;      // device: [num_sentences]
    int               num_sentences;
    float             base_timestamp;
    NMEAParsed*       d_parsed;       // device: [num_sentences] — output

    // Stage 1: Kalman predict
    NavState*         d_states;       // device: [num_states] — in/out
    CovarianceFP16*   d_covariances;  // device: [num_states] — in/out
    int               num_states;
    float             dt;             // seconds since last predict

    // Stage 2: Sonar waterfall
    const float*      d_raw_returns;  // device: [num_pings * samples_per_ping]
    float*            d_waterfall;    // device: [num_pings * waterfall_width] — output
    int               samples_per_ping;
    int               waterfall_width;
    float             time_per_sample; // seconds (e.g. 1/44100)
    float             sound_velocity;  // m/s (seawater ~1500)
    int               num_pings;

    // Stage 3: Constraint check
    const NavConstraint* d_constraints; // device: [num_constraints]
    ConstraintResult*    d_results;     // device: [num_constraints * num_states] — output
    int                  num_constraints;
};

// Per-stage kernel timing reported by run_fusion_pipeline() when timing != nullptr.
struct FusionTiming {
    float nmea_ms;        // Stage 0
    float kalman_ms;      // Stage 1
    float sonar_ms;       // Stage 2
    float constraint_ms;  // Stage 3
    float total_ms;       // wall time start → end (includes event overhead)
};
