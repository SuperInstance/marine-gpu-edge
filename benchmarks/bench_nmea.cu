/**
 * bench_nmea.cu — NMEA parse throughput benchmark
 *
 * Generates 100K synthetic GGA sentences, parses them on the GPU, and
 * compares against a CPU reference implementation.
 *
 * Reports:
 *   - GPU kernel throughput  (sentences/sec, kernel time only)
 *   - GPU end-to-end throughput (includes H2D + D2H transfers)
 *   - CPU throughput
 *   - GPU vs CPU speedup
 *
 * Usage:  ./bench_nmea [num_sentences]
 *         Default: 100000
 *
 * Author: Forgemaster ⚒️
 * License: MIT
 */

#include "marine_types.h"

#include <cuda_runtime.h>
#include <cstdio>

// Forward declaration — implementation in marine_sensor_fusion.cu
extern "C" cudaError_t marine_parse_nmea(
    const char* sentences, const int* lengths,
    int num_sentences, NMEAParsed* output,
    float base_timestamp, cudaStream_t stream);
#include <cstdlib>
#include <cstring>
#include <cmath>
#include <ctime>
#include <cassert>

// ============================================================
// SYNTHETIC NMEA GENERATOR
// ============================================================

/** Write one synthetic GGA sentence into `buf` (exactly NMEA_SENTENCE_STRIDE bytes). */
static void generate_gga(char* buf, int idx) {
    // Vary position slightly so the compiler can't eliminate the work
    float lat_deg = 37.0f + (float)(idx % 100) * 0.001f;
    float lon_deg = -122.0f + (float)(idx % 137) * 0.001f;

    // Format: DDMM.MMMM
    int   lat_d  = (int)lat_deg;
    float lat_m  = (lat_deg - lat_d) * 60.0f;
    int   lon_d  = (int)(-lon_deg);
    float lon_m  = ((-lon_deg) - lon_d) * 60.0f;

    int hh = (idx / 3600) % 24;
    int mm = (idx /   60) % 60;
    int ss =  idx         % 60;

    // Build sentence body (between $ and *)
    char body[NMEA_SENTENCE_STRIDE];
    int  blen = snprintf(body, sizeof(body),
        "GPGGA,%02d%02d%02d.00,%02d%07.4f,N,%03d%07.4f,W,1,08,1.0,5.0,M,0.0,M,,",
        hh, mm, ss,
        lat_d, (double)lat_m,
        lon_d, (double)lon_m
    );

    // Compute NMEA checksum (XOR of bytes between $ and *)
    uint8_t cksum = 0;
    for (int i = 0; i < blen; i++) cksum ^= (uint8_t)body[i];

    // Assemble full sentence
    int total = snprintf(buf, NMEA_SENTENCE_STRIDE,
        "$%s*%02X\r\n", body, cksum);

    // Pad with NUL to fill stride
    memset(buf + total, 0, NMEA_SENTENCE_STRIDE - total);
}

// ============================================================
// CPU REFERENCE IMPLEMENTATION
// ============================================================

static uint8_t cpu_nmea_checksum(const char* s, int len) {
    uint8_t ck = 0;
    for (int i = 1; i < len && s[i] != '*'; i++) ck ^= (uint8_t)s[i];
    return ck;
}

static int cpu_hex_to_int(char c) {
    if (c >= '0' && c <= '9') return c - '0';
    if (c >= 'A' && c <= 'F') return c - 'A' + 10;
    if (c >= 'a' && c <= 'f') return c - 'a' + 10;
    return -1;
}

static void cpu_parse_nmea(
    const char* sentences,
    const int*  lengths,
    int         num,
    NMEAParsed* output,
    float       base_ts
) {
    for (int idx = 0; idx < num; idx++) {
        const char* s   = sentences + (size_t)idx * NMEA_SENTENCE_STRIDE;
        int         len = lengths[idx];

        NMEAParsed r    = {};
        r.timestamp     = base_ts + (float)idx * 0.1f;

        // Checksum
        uint8_t computed = cpu_nmea_checksum(s, len);
        int star_pos = -1;
        for (int i = 0; i < len; i++) {
            if (s[i] == '*') { star_pos = i; break; }
        }
        if (star_pos > 0 && star_pos + 2 < len) {
            int hi = cpu_hex_to_int(s[star_pos + 1]);
            int lo = cpu_hex_to_int(s[star_pos + 2]);
            if (hi >= 0 && lo >= 0)
                r.valid = ((uint8_t)((hi << 4) | lo) == computed) ? 1 : 0;
        }

        // Type
        if (len > 6) {
            char c3 = s[3], c4 = s[4], c5 = s[5];
            if      (c3=='G'&&c4=='G'&&c5=='A') r.sentence_type = 1;
            else if (c3=='R'&&c4=='M'&&c5=='C') r.sentence_type = 2;
            else if (c3=='H'&&c4=='D'&&c5=='T') r.sentence_type = 3;
        }

        output[idx] = r;
    }
}

// ============================================================
// TIMING HELPER
// ============================================================

static double now_sec() {
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return ts.tv_sec + ts.tv_nsec * 1e-9;
}

// ============================================================
// MAIN
// ============================================================

int main(int argc, char** argv) {
    int N = 100000;
    if (argc > 1) N = atoi(argv[1]);
    if (N <= 0 || N > 10000000) {
        fprintf(stderr, "num_sentences must be in [1, 10000000]\n");
        return 1;
    }

    // Detect GPU
    int dev = 0;
    cudaDeviceProp prop;
    cudaGetDeviceProperties(&prop, dev);
    printf("=== bench_nmea: NMEA parse throughput ===\n");
    printf("GPU : %s (SM %d.%d)\n", prop.name, prop.major, prop.minor);
    printf("N   : %d sentences\n\n", N);

    // ---- Allocate host buffers ----
    size_t stride_bytes  = (size_t)N * NMEA_SENTENCE_STRIDE;
    char*  h_sentences   = (char*)malloc(stride_bytes);
    int*   h_lengths     = (int*) malloc(N * sizeof(int));
    NMEAParsed* h_result = (NMEAParsed*)malloc(N * sizeof(NMEAParsed));
    NMEAParsed* h_cpu    = (NMEAParsed*)malloc(N * sizeof(NMEAParsed));
    if (!h_sentences || !h_lengths || !h_result || !h_cpu) {
        fprintf(stderr, "Host malloc failed\n"); return 1;
    }

    // Generate sentences
    printf("Generating %d synthetic GGA sentences...\n", N);
    for (int i = 0; i < N; i++) {
        generate_gga(h_sentences + (size_t)i * NMEA_SENTENCE_STRIDE, i);
        // Measure actual length (up to \n or stride)
        const char* s = h_sentences + (size_t)i * NMEA_SENTENCE_STRIDE;
        int len = 0;
        while (len < NMEA_SENTENCE_STRIDE && s[len] != '\0') len++;
        h_lengths[i] = len;
    }
    printf("Done.\n\n");

    // ---- GPU BENCHMARK ----

    // Allocate device buffers
    char*       d_sentences = nullptr;
    int*        d_lengths   = nullptr;
    NMEAParsed* d_result    = nullptr;

    cudaMalloc(&d_sentences, stride_bytes);
    cudaMalloc(&d_lengths,   N * sizeof(int));
    cudaMalloc(&d_result,    N * sizeof(NMEAParsed));

    // CUDA events for kernel-only timing
    cudaEvent_t ev_start, ev_stop;
    cudaEventCreate(&ev_start);
    cudaEventCreate(&ev_stop);

    // Warm-up pass (avoid cold-start effects)
    cudaMemcpy(d_sentences, h_sentences, stride_bytes,      cudaMemcpyHostToDevice);
    cudaMemcpy(d_lengths,   h_lengths,   N * sizeof(int),   cudaMemcpyHostToDevice);
    marine_parse_nmea(d_sentences, d_lengths, N, d_result, 0.0f, 0);
    cudaDeviceSynchronize();

    // End-to-end timing (H2D + kernel + D2H)
    double e2e_start = now_sec();

    cudaMemcpy(d_sentences, h_sentences, stride_bytes,    cudaMemcpyHostToDevice);
    cudaMemcpy(d_lengths,   h_lengths,   N * sizeof(int), cudaMemcpyHostToDevice);

    // Kernel-only timing
    cudaEventRecord(ev_start);
    marine_parse_nmea(d_sentences, d_lengths, N, d_result, 0.0f, 0);
    cudaEventRecord(ev_stop);

    cudaMemcpy(h_result, d_result, N * sizeof(NMEAParsed), cudaMemcpyDeviceToHost);
    cudaDeviceSynchronize();

    double e2e_end = now_sec();
    float  kernel_ms;
    cudaEventElapsedTime(&kernel_ms, ev_start, ev_stop);

    double gpu_kernel_throughput = (double)N / (kernel_ms * 1e-3);
    double gpu_e2e_throughput    = (double)N / (e2e_end - e2e_start);

    // ---- CPU BENCHMARK ----

    // Single run for reference
    double cpu_start = now_sec();
    cpu_parse_nmea(h_sentences, h_lengths, N, h_cpu, 0.0f);
    double cpu_end = now_sec();

    double cpu_throughput = (double)N / (cpu_end - cpu_start);

    // ---- VALIDATION ----
    // Spot-check: valid flags and sentence types should match
    int mismatches = 0;
    for (int i = 0; i < N; i++) {
        if (h_result[i].valid         != h_cpu[i].valid ||
            h_result[i].sentence_type != h_cpu[i].sentence_type) {
            mismatches++;
        }
    }

    // ---- REPORT ----
    printf("--- GPU (kernel only) ---\n");
    printf("  Kernel time  : %.3f ms\n", kernel_ms);
    printf("  Throughput   : %.2f M sentences/sec\n", gpu_kernel_throughput / 1e6);

    printf("\n--- GPU (end-to-end, incl. H2D+D2H) ---\n");
    printf("  Total time   : %.3f ms\n", (e2e_end - e2e_start) * 1e3);
    printf("  Throughput   : %.2f M sentences/sec\n", gpu_e2e_throughput / 1e6);

    printf("\n--- CPU reference ---\n");
    printf("  Time         : %.3f ms\n", (cpu_end - cpu_start) * 1e3);
    printf("  Throughput   : %.2f M sentences/sec\n", cpu_throughput / 1e6);

    printf("\n--- Comparison ---\n");
    printf("  GPU kernel speedup vs CPU : %.1fx\n", gpu_kernel_throughput / cpu_throughput);
    printf("  GPU e2e speedup vs CPU    : %.1fx\n", gpu_e2e_throughput    / cpu_throughput);

    if (mismatches > 0) {
        printf("\n  WARNING: %d/%d result mismatches (valid/type)\n", mismatches, N);
    } else {
        printf("\n  Validation: PASS (%d/%d sentences match CPU reference)\n", N, N);
    }

    // ---- CLEANUP ----
    cudaFree(d_sentences);
    cudaFree(d_lengths);
    cudaFree(d_result);
    cudaEventDestroy(ev_start);
    cudaEventDestroy(ev_stop);
    free(h_sentences);
    free(h_lengths);
    free(h_result);
    free(h_cpu);

    return (mismatches == 0) ? 0 : 1;
}
