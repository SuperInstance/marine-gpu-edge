/**
 * fusion_pipeline.cu — Multi-stage GPU fusion pipeline orchestrator
 *
 * Chains NMEA parse → Kalman predict → sonar waterfall → constraint check
 * into a single CUDA stream with event-based profiling.
 *
 * Novel: Supports both fused (single device) and split (workstation + edge)
 * pipeline modes, plus cudaGraph capture/replay for zero-launch-overhead.
 *
 * Author: Forgemaster ⚒️
 * License: MIT
 */

#include "marine_types.h"
#include <cuda_runtime.h>
#include <cstdio>
#include <cstring>

// Forward declarations from marine_sensor_fusion.cu
extern "C" cudaError_t marine_parse_nmea(
    const char* sentences, const int* lengths,
    int num_sentences, NMEAParsed* output,
    float base_timestamp, cudaStream_t stream);

extern "C" cudaError_t marine_kalman_predict(
    NavState* states, CovarianceFP16* covariances,
    float dt, int count, cudaStream_t stream);

extern "C" cudaError_t marine_sonar_waterfall(
    const float* raw_returns, float* waterfall,
    int samples_per_ping, int waterfall_width,
    float time_per_sample, float sound_velocity,
    int num_pings, cudaStream_t stream);

extern "C" cudaError_t marine_check_constraints(
    const NavState* states, const NavConstraint* constraints,
    ConstraintResult* results, int num_constraints,
    int num_states, cudaStream_t stream);

// ============================================================
// Pipeline runner — fused single-device mode
// ============================================================

cudaError_t run_fusion_pipeline(
    // Inputs (device pointers)
    const char*         d_sentences,
    const int*          d_lengths,
    const float*        d_sonar_raw,
    const NavConstraint* d_constraints,

    // Outputs (device pointers, pre-allocated by caller)
    NMEAParsed*         d_nmea_out,
    NavState*           d_nav_states,
    CovarianceFP16*     d_covariances,
    float*              d_waterfall,
    ConstraintResult*   d_constraint_results,

    const FusionConfig& config,
    FusionResult&       result,
    cudaStream_t        stream
) {
    cudaEvent_t start, stop, stage_start, stage_stop;
    cudaError_t err;

    err = cudaEventCreate(&start);
    if (err != cudaSuccess) return err;
    err = cudaEventCreate(&stop);
    if (err != cudaSuccess) return err;
    err = cudaEventCreate(&stage_start);
    if (err != cudaSuccess) return err;
    err = cudaEventCreate(&stage_stop);
    if (err != cudaSuccess) return err;

    err = cudaEventRecord(start, stream);
    if (err != cudaSuccess) return err;

    memset(&result, 0, sizeof(result));

    // --- Stage 0: NMEA Parse ---
    if (config.enable_nmea_parse && d_sentences && d_nmea_out) {
        err = cudaEventRecord(stage_start, stream);
        if (err != cudaSuccess) return err;

        err = marine_parse_nmea(d_sentences, d_lengths,
            config.num_sentences, d_nmea_out,
            config.base_timestamp, stream);
        if (err != cudaSuccess) return err;

        err = cudaEventRecord(stage_stop, stream);
        if (err != cudaSuccess) return err;
        err = cudaEventSynchronize(stage_stop);
        if (err != cudaSuccess) return err;
        err = cudaEventElapsedTime(&result.time_nmea, stage_start, stage_stop);
        if (err != cudaSuccess) return err;

        result.nmea_per_sec = config.num_sentences / (result.time_nmea * 0.001f);
    }

    // --- Stage 1: Kalman Predict ---
    if (config.enable_kalman && d_nav_states && d_covariances) {
        err = cudaEventRecord(stage_start, stream);
        if (err != cudaSuccess) return err;

        for (int step = 0; step < config.kalman_steps; step++) {
            err = marine_kalman_predict(d_nav_states, d_covariances,
                config.dt, config.num_sentences, stream);
            if (err != cudaSuccess) return err;
        }

        err = cudaEventRecord(stage_stop, stream);
        if (err != cudaSuccess) return err;
        err = cudaEventSynchronize(stage_stop);
        if (err != cudaSuccess) return err;
        err = cudaEventElapsedTime(&result.time_kalman, stage_start, stage_stop);
        if (err != cudaSuccess) return err;

        result.kalman_per_sec = config.num_sentences * config.kalman_steps
                                / (result.time_kalman * 0.001f);
    }

    // --- Stage 2: Sonar Waterfall ---
    if (config.enable_sonar && d_sonar_raw && d_waterfall) {
        err = cudaEventRecord(stage_start, stream);
        if (err != cudaSuccess) return err;

        err = marine_sonar_waterfall(d_sonar_raw, d_waterfall,
            config.samples_per_ping, config.waterfall_width,
            config.time_per_sample, config.sound_velocity,
            config.num_sentences, stream);
        if (err != cudaSuccess) return err;

        err = cudaEventRecord(stage_stop, stream);
        if (err != cudaSuccess) return err;
        err = cudaEventSynchronize(stage_stop);
        if (err != cudaSuccess) return err;
        err = cudaEventElapsedTime(&result.time_sonar, stage_start, stage_stop);
        if (err != cudaSuccess) return err;

        result.pings_per_sec = config.num_sentences / (result.time_sonar * 0.001f);
    }

    // --- Stage 3: Constraint Check ---
    if (config.enable_constraints && d_nav_states && d_constraints
        && d_constraint_results) {
        err = cudaEventRecord(stage_start, stream);
        if (err != cudaSuccess) return err;

        err = marine_check_constraints(d_nav_states, d_constraints,
            d_constraint_results, config.num_constraints,
            config.num_sentences, stream);
        if (err != cudaSuccess) return err;

        err = cudaEventRecord(stage_stop, stream);
        if (err != cudaSuccess) return err;
        err = cudaEventSynchronize(stage_stop);
        if (err != cudaSuccess) return err;
        err = cudaEventElapsedTime(&result.time_constr, stage_start, stage_stop);
        if (err != cudaSuccess) return err;

        // Count violations on host
        ConstraintResult* h_results = (ConstraintResult*)malloc(
            config.num_sentences * sizeof(ConstraintResult));
        if (h_results) {
            err = cudaMemcpyAsync(h_results, d_constraint_results,
                config.num_sentences * sizeof(ConstraintResult),
                cudaMemcpyDeviceToHost, stream);
            if (err == cudaSuccess) {
                err = cudaStreamSynchronize(stream);
                if (err == cudaSuccess) {
                    for (int i = 0; i < config.num_sentences; i++) {
                        if (h_results[i].violated) {
                            result.violations_found++;
                            if (h_results[i].severity >= 2)
                                result.critical_violations++;
                        }
                    }
                }
            }
            free(h_results);
        }
    }

    // --- Total ---
    err = cudaEventRecord(stop, stream);
    if (err != cudaSuccess) return err;
    err = cudaEventSynchronize(stop);
    if (err != cudaSuccess) return err;
    err = cudaEventElapsedTime(&result.time_total, start, stop);
    if (err != cudaSuccess) return err;

    // Cleanup events
    cudaEventDestroy(start);
    cudaEventDestroy(stop);
    cudaEventDestroy(stage_start);
    cudaEventDestroy(stage_stop);

    return cudaSuccess;
}

// ============================================================
// cudaGraph capture/replay — zero-launch-overhead pipeline
// ============================================================

cudaError_t capture_fusion_graph(
    cudaGraph_t& graph,
    // Inputs (device pointers)
    const char*         d_sentences,
    const int*          d_lengths,
    const float*        d_sonar_raw,
    const NavConstraint* d_constraints,
    // Outputs (device pointers)
    NMEAParsed*         d_nmea_out,
    NavState*           d_nav_states,
    CovarianceFP16*     d_covariances,
    float*              d_waterfall,
    ConstraintResult*   d_constraint_results,
    const FusionConfig& config,
    cudaStream_t        stream
) {
    // Capture the pipeline into a graph
    cudaError_t err = cudaStreamBeginCapture(stream, cudaStreamCaptureModeGlobal);
    if (err != cudaSuccess) return err;

    // Run the pipeline (no profiling — graph capture records kernel launches)
    if (config.enable_nmea_parse && d_sentences && d_nmea_out) {
        marine_parse_nmea(d_sentences, d_lengths,
            config.num_sentences, d_nmea_out,
            config.base_timestamp, stream);
    }

    if (config.enable_kalman && d_nav_states && d_covariances) {
        for (int step = 0; step < config.kalman_steps; step++) {
            marine_kalman_predict(d_nav_states, d_covariances,
                config.dt, config.num_sentences, stream);
        }
    }

    if (config.enable_sonar && d_sonar_raw && d_waterfall) {
        marine_sonar_waterfall(d_sonar_raw, d_waterfall,
            config.samples_per_ping, config.waterfall_width,
            config.time_per_sample, config.sound_velocity,
            config.num_sentences, stream);
    }

    if (config.enable_constraints && d_nav_states && d_constraints
        && d_constraint_results) {
        marine_check_constraints(d_nav_states, d_constraints,
            d_constraint_results, config.num_constraints,
            config.num_sentences, stream);
    }

    err = cudaStreamEndCapture(stream, &graph);
    return err;
}

cudaError_t replay_fusion_graph(
    cudaGraphExec_t& graph_exec,
    cudaGraph_t&     graph,
    cudaStream_t     stream
) {
    cudaError_t err;

    if (!graph_exec) {
        err = cudaGraphInstantiate(&graph_exec, graph, nullptr, nullptr, 0);
        if (err != cudaSuccess) return err;
    }

    err = cudaGraphLaunch(graph_exec, stream);
    return err;
}
