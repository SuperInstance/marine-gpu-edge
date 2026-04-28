/**
 * bench_fusion.cu — Full integration benchmark for the fusion pipeline
 *
 * Exercises: NMEA parse → Kalman predict → sonar waterfall → constraint check
 * Generates synthetic data, runs fused pipeline, reports per-stage timing.
 *
 * Author: Forgemaster ⚒️
 * License: MIT
 */

#include "marine_types.h"
#include <cuda_runtime.h>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <cmath>
#include <ctime>

// Pipeline host functions (from fusion_pipeline.cu)
extern cudaError_t run_fusion_pipeline(
    const char*, const int*, const float*, const NavConstraint*,
    NMEAParsed*, NavState*, CovarianceFP16*, float*, ConstraintResult*,
    const FusionConfig&, FusionResult&, cudaStream_t);

extern cudaError_t capture_fusion_graph(cudaGraph_t&,
    const char*, const int*, const float*, const NavConstraint*,
    NMEAParsed*, NavState*, CovarianceFP16*, float*, ConstraintResult*,
    const FusionConfig&, cudaStream_t);

extern cudaError_t replay_fusion_graph(cudaGraphExec_t&, cudaGraph_t&,
    cudaStream_t);

// ============================================================
// Synthetic data generators
// ============================================================

// Generate NMEA GGA sentences around Anchorage AK (61.2°N, -149.9°W)
static void generate_nmea_gga(char* buf, int stride, int count) {
    // Anchorage: 6121.5270 N, 14954.9180 W
    for (int i = 0; i < count; i++) {
        char* s = buf + (size_t)i * stride;
        // Add slight random variation
        float lat_var = ((float)(i % 1000) - 500.0f) * 0.0001f;
        float lon_var = ((float)((i * 7) % 1000) - 500.0f) * 0.0001f;
        double lat_min = 21.5270 + lat_var;
        double lon_min = 54.9180 + lon_var;

        int hh = (i * 3) % 24;
        int mm = (i * 7) % 60;
        int ss = (i * 11) % 60;
        int fix = (i > 10) ? 1 : 0;  // first 10 have no fix
        int nsat = 8 + (i % 7);

        snprintf(s, stride,
            "$GPGGA,%02d%02d%02d.00,6121.%04.4f,N,14954.%04.4f,W,%d,%02d,1.2,25.6,M,-17.8,M,,*",
            hh, mm, ss, lat_min, lon_min, fix, nsat);

        // Compute and append checksum
        int len = strlen(s);
        uint8_t cksum = 0;
        for (int j = 1; j < len && s[j] != '*'; j++) cksum ^= (uint8_t)s[j];
        char hex[3];
        snprintf(hex, sizeof(hex), "%02X", cksum);
        s[len] = hex[0]; s[len+1] = hex[1]; s[len+2] = '\0';

        // Pad rest with NUL
        for (int j = len + 3; j < stride; j++) s[j] = '\0';
    }
}

// Generate NavState array (FP16, around Anchorage)
static void generate_nav_states(NavState* states, int count) {
    for (int i = 0; i < count; i++) {
        states[i].pos.x  = __float2half(61.2f + ((float)(i % 1000) - 500.0f) * 0.0001f);
        states[i].pos.y  = __float2half(-149.9f + ((float)((i * 7) % 1000) - 500.0f) * 0.0001f);
        states[i].vel.x  = __float2half(1.5f + ((float)(i % 10) * 0.1f));  // ~3 knots north
        states[i].vel.y  = __float2half(0.8f + ((float)((i * 3) % 10) * 0.1f));
        states[i].heading = __float2half((float)((i * 13) % 360) * (float)M_PI / 180.0f);
        states[i].heading_rate = __float2half(0.01f);
        states[i].depth   = 12.0f + (float)(i % 20) * 0.5f;  // 12-22m depth
        states[i].timestamp = (float)i * 0.1f;
    }
}

// Generate NavConstraint array
static void generate_constraints(NavConstraint* constraints, int count) {
    memset(constraints, 0, count * sizeof(NavConstraint));
    // Constraint 0: minimum depth 10m
    constraints[0].active = 1;
    constraints[0].min_depth = 10.0f;
    // Constraint 1: max speed 15 knots
    constraints[1].active = 1;
    constraints[1].max_speed = 15.0f;
    // Constraint 2: max speed 25 knots
    constraints[2].active = 1;
    constraints[2].max_speed = 25.0f;
    // Constraint 3: safety radius 100m
    constraints[3].active = 1;
    constraints[3].safety_radius = 100.0f;
    // Constraint 4: minimum depth 15m
    constraints[4].active = 1;
    constraints[4].min_depth = 15.0f;
    // Constraint 5: max speed 6 knots (no-wake zone)
    constraints[5].active = 1;
    constraints[5].max_speed = 6.0f;
    // Constraint 6: max roll 30 degrees
    constraints[6].active = 1;
    constraints[6].max_roll = 30.0f;
    // Constraint 7: minimum depth 20m
    constraints[7].active = 1;
    constraints[7].min_depth = 20.0f;
}

// Generate synthetic sonar returns (Gaussian noise + simulated bottom)
static void generate_sonar_returns(float* data, int num_pings, int samples_per_ping) {
    srand(42);
    for (int p = 0; p < num_pings; p++) {
        float bottom_sample = samples_per_ping * (0.5f + 0.1f * sinf((float)p * 0.01f));
        for (int s = 0; s < samples_per_ping; s++) {
            float noise = ((float)rand() / (float)RAND_MAX - 0.5f) * 0.2f;
            if (s < bottom_sample - 50) {
                // Water column: mostly noise
                data[p * samples_per_ping + s] = noise * 0.1f;
            } else if (s < bottom_sample + 50) {
                // Bottom return: strong signal
                float dist = fabsf((float)s - bottom_sample);
                data[p * samples_per_ping + s] = 0.8f * expf(-dist * dist / 500.0f) + noise * 0.3f;
            } else {
                // Below bottom: noise
                data[p * samples_per_ping + s] = noise * 0.05f;
            }
        }
    }
}

// ============================================================
// Main benchmark
// ============================================================

int main() {
    printf("=== bench_fusion: Full Integration Benchmark ===\n\n");

    cudaDeviceProp prop;
    cudaGetDeviceProperties(&prop, 0);
    printf("GPU: %s (SM %d.%d)\n\n", prop.name, prop.major, prop.minor);

    const int N = 10000;
    const int SENTENCE_STRIDE = 128;
    const int SPP = 2048;       // samples per ping
    const int WF_WIDTH = 512;   // waterfall bins
    const int N_CONSTR = 8;

    FusionConfig config;
    config.num_sentences    = N;
    config.samples_per_ping = SPP;
    config.waterfall_width  = WF_WIDTH;
    config.num_constraints  = N_CONSTR;
    config.dt               = 0.1f;
    config.kalman_steps     = 1;

    // --- Allocate host memory ---
    char* h_sentences = (char*)calloc(N * SENTENCE_STRIDE, 1);
    int*  h_lengths   = (int*)calloc(N, sizeof(int));
    NavState* h_states = (NavState*)calloc(N, sizeof(NavState));
    CovarianceFP16* h_covs = (CovarianceFP16*)calloc(N, sizeof(CovarianceFP16));
    NavConstraint* h_constrs = (NavConstraint*)calloc(N_CONSTR, sizeof(NavConstraint));
    float* h_sonar = (float*)calloc(N * SPP, sizeof(float));

    if (!h_sentences || !h_lengths || !h_states || !h_covs || !h_constrs || !h_sonar) {
        fprintf(stderr, "Host alloc failed\n");
        return 1;
    }

    // --- Generate synthetic data ---
    printf("Generating %d NMEA sentences...\n", N);
    generate_nmea_gga(h_sentences, SENTENCE_STRIDE, N);
    for (int i = 0; i < N; i++) h_lengths[i] = strlen(h_sentences + (size_t)i * SENTENCE_STRIDE);

    printf("Generating %d nav states around Anchorage AK...\n", N);
    generate_nav_states(h_states, N);

    // Initialize covariances (identity-ish)
    for (int i = 0; i < N; i++) {
        memset(&h_covs[i], 0, sizeof(CovarianceFP16));
        h_covs[i].data[0] = __float2half(0.01f);   // lat var
        h_covs[i].data[2] = __float2half(0.01f);   // lon var
        h_covs[i].data[5] = __float2half(0.1f);    // vn var
        h_covs[i].data[9] = __float2half(0.1f);    // ve var
        h_covs[i].data[14] = __float2half(0.05f);  // heading var
    }

    printf("Generating %d navigation constraints...\n", N_CONSTR);
    generate_constraints(h_constrs, N_CONSTR);

    printf("Generating %d sonar pings (%d samples each)...\n", N, SPP);
    generate_sonar_returns(h_sonar, N, SPP);

    // --- Allocate device memory ---
    char* d_sentences; int* d_lengths;
    NMEAParsed* d_nmea_out; NavState* d_states; CovarianceFP16* d_covs;
    float* d_sonar; float* d_waterfall;
    NavConstraint* d_constrs; ConstraintResult* d_constr_results;

    cudaMalloc(&d_sentences, N * SENTENCE_STRIDE);
    cudaMalloc(&d_lengths, N * sizeof(int));
    cudaMalloc(&d_nmea_out, N * sizeof(NMEAParsed));
    cudaMalloc(&d_states, N * sizeof(NavState));
    cudaMalloc(&d_covs, N * sizeof(CovarianceFP16));
    cudaMalloc(&d_sonar, N * SPP * sizeof(float));
    cudaMalloc(&d_waterfall, N * WF_WIDTH * sizeof(float));
    cudaMalloc(&d_constrs, N_CONSTR * sizeof(NavConstraint));
    cudaMalloc(&d_constr_results, N * sizeof(ConstraintResult));

    // --- H2D transfers ---
    cudaMemcpy(d_sentences, h_sentences, N * SENTENCE_STRIDE, cudaMemcpyHostToDevice);
    cudaMemcpy(d_lengths, h_lengths, N * sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(d_states, h_states, N * sizeof(NavState), cudaMemcpyHostToDevice);
    cudaMemcpy(d_covs, h_covs, N * sizeof(CovarianceFP16), cudaMemcpyHostToDevice);
    cudaMemcpy(d_sonar, h_sonar, N * SPP * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_constrs, h_constrs, N_CONSTR * sizeof(NavConstraint), cudaMemcpyHostToDevice);
    cudaMemset(d_nmea_out, 0, N * sizeof(NMEAParsed));
    cudaMemset(d_waterfall, 0, N * WF_WIDTH * sizeof(float));
    cudaMemset(d_constr_results, 0, N * sizeof(ConstraintResult));

    // --- Run fusion pipeline ---
    printf("\n--- Fused Pipeline (all stages, single stream) ---\n");

    cudaStream_t stream;
    cudaStreamCreate(&stream);

    FusionResult result;
    cudaError_t err = run_fusion_pipeline(
        d_sentences, d_lengths, d_sonar, d_constrs,
        d_nmea_out, d_states, d_covs, d_waterfall, d_constr_results,
        config, result, stream);

    if (err != cudaSuccess) {
        fprintf(stderr, "Pipeline failed: %s\n", cudaGetErrorString(err));
        return 1;
    }

    printf("  NMEA parse    : %8.3f ms  (%.1f M sent/sec)\n",
        result.time_nmea, result.nmea_per_sec / 1e6f);
    printf("  Kalman predict: %8.3f ms  (%.1f M states/sec)\n",
        result.time_kalman, result.kalman_per_sec / 1e6f);
    printf("  Sonar waterfall: %8.3f ms  (%.1f K pings/sec)\n",
        result.time_sonar, result.pings_per_sec / 1e3f);
    printf("  Constraint chk: %8.3f ms\n", result.time_constr);
    printf("  TOTAL         : %8.3f ms\n", result.time_total);
    printf("\n  Constraint violations: %d total, %d critical\n",
        result.violations_found, result.critical_violations);

    // --- cudaGraph capture/replay benchmark ---
    printf("\n--- cudaGraph Capture & Replay ---\n");

    cudaGraph_t graph = nullptr;
    cudaGraphExec_t graph_exec = nullptr;

    err = capture_fusion_graph(graph,
        d_sentences, d_lengths, d_sonar, d_constrs,
        d_nmea_out, d_states, d_covs, d_waterfall, d_constr_results,
        config, stream);

    if (err == cudaSuccess) {
        // Warmup replay
        replay_fusion_graph(graph_exec, graph, stream);
        cudaStreamSynchronize(stream);

        // Benchmark 100 replays
        cudaEvent_t g_start, g_stop;
        cudaEventCreate(&g_start);
        cudaEventCreate(&g_stop);

        cudaEventRecord(g_start, stream);
        for (int r = 0; r < 100; r++) {
            replay_fusion_graph(graph_exec, graph, stream);
        }
        cudaEventRecord(g_stop, stream);
        cudaStreamSynchronize(stream);

        float graph_ms;
        cudaEventElapsedTime(&graph_ms, g_start, g_stop);
        float per_iter_ms = graph_ms / 100.0f;

        printf("  100 replays  : %.3f ms total\n", graph_ms);
        printf("  Per replay   : %.3f ms\n", per_iter_ms);
        printf("  vs stream    : %.2fx speedup\n", result.time_total / per_iter_ms);

        cudaEventDestroy(g_start);
        cudaEventDestroy(g_stop);
    } else {
        printf("  Graph capture failed: %s\n", cudaGetErrorString(err));
    }

    if (graph_exec) cudaGraphExecDestroy(graph_exec);
    if (graph) cudaGraphDestroy(graph);

    // --- Stage-only benchmarks ---
    printf("\n--- Stage Isolation (5 warmup + 50 measured) ---\n");

    // NMEA-only
    {
        float total = 0;
        for (int r = 0; r < 55; r++) {
            FusionResult sr;
            FusionConfig sc = config;
            sc.enable_kalman = sc.enable_sonar = sc.enable_constraints = false;
            cudaEvent_t s0, s1;
            cudaEventCreate(&s0); cudaEventCreate(&s1);
            cudaEventRecord(s0, stream);
            marine_parse_nmea(d_sentences, d_lengths, N, d_nmea_out, 0.0f, stream);
            cudaEventRecord(s1, stream);
            cudaStreamSynchronize(stream);
            float ms;
            cudaEventElapsedTime(&ms, s0, s1);
            cudaEventDestroy(s0); cudaEventDestroy(s1);
            if (r >= 5) total += ms;
        }
        printf("  NMEA (avg)   : %.3f ms  (%.1f M/sec)\n", total/50, N/(total/50*0.001f)/1e6f);
    }

    // Kalman-only
    {
        float total = 0;
        for (int r = 0; r < 55; r++) {
            cudaEvent_t s0, s1;
            cudaEventCreate(&s0); cudaEventCreate(&s1);
            cudaEventRecord(s0, stream);
            marine_kalman_predict(d_states, d_covs, 0.1f, N, stream);
            cudaEventRecord(s1, stream);
            cudaStreamSynchronize(stream);
            float ms;
            cudaEventElapsedTime(&ms, s0, s1);
            cudaEventDestroy(s0); cudaEventDestroy(s1);
            if (r >= 5) total += ms;
        }
        printf("  Kalman (avg) : %.3f ms  (%.1f M states/sec)\n", total/50, N/(total/50*0.001f)/1e6f);
    }

    // Sonar-only
    {
        float total = 0;
        for (int r = 0; r < 55; r++) {
            cudaEvent_t s0, s1;
            cudaEventCreate(&s0); cudaEventCreate(&s1);
            cudaEventRecord(s0, stream);
            marine_sonar_waterfall(d_sonar, d_waterfall, SPP, WF_WIDTH,
                0.00005f, 1500.0f, N, stream);
            cudaEventRecord(s1, stream);
            cudaStreamSynchronize(stream);
            float ms;
            cudaEventElapsedTime(&ms, s0, s1);
            cudaEventDestroy(s0); cudaEventDestroy(s1);
            if (r >= 5) total += ms;
        }
        printf("  Sonar (avg)  : %.3f ms  (%.1f K pings/sec)\n", total/50, N/(total/50*0.001f)/1e3f);
    }

    printf("\n=== Benchmark Complete ===\n");

    // Cleanup
    cudaStreamDestroy(stream);
    cudaFree(d_sentences); cudaFree(d_lengths);
    cudaFree(d_nmea_out); cudaFree(d_states); cudaFree(d_covs);
    cudaFree(d_sonar); cudaFree(d_waterfall);
    cudaFree(d_constrs); cudaFree(d_constr_results);
    free(h_sentences); free(h_lengths); free(h_states);
    free(h_covs); free(h_constrs); free(h_sonar);

    return 0;
}
