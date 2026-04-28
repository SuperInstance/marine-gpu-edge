/**
 * bench_beamformer.cu — Multibeam sonar beamforming benchmark
 *
 * Generates synthetic 32-element array data, runs delay-and-sum beamforming
 * on GPU, measures throughput and validates beam peak detection.
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

extern "C" cudaError_t run_beamformer(
    const float*, float*, const float*,
    const BeamformerConfig&, cudaStream_t, float&);

extern "C" void generate_multibeam_data(
    float*, const BeamformerConfig&, int);

static void build_delay_table(float*, const BeamformerConfig&);

int main() {
    printf("=== bench_beamformer: Multibeam Sonar Beamforming ===\n\n");

    cudaDeviceProp prop;
    cudaGetDeviceProperties(&prop, 0);
    printf("GPU: %s (SM %d.%d)\n", prop.name, prop.major, prop.minor);
    printf("Shared memory per block: %zu KB\n\n", prop.sharedMemPerBlock / 1024);

    BeamformerConfig config;
    config.num_elements = 32;
    config.num_beams = 128;
    config.num_samples = 2048;
    config.element_spacing = 0.02f;
    config.sound_velocity = 1500.0f;
    config.sample_rate = 40000.0f;
    config.beam_width_deg = 180.0f;

    size_t channel_bytes = (size_t)config.num_elements * config.num_samples * sizeof(float);
    size_t beam_bytes = (size_t)config.num_beams * config.num_samples * sizeof(float);
    size_t delay_bytes = (size_t)config.num_beams * config.num_elements * sizeof(float);

    printf("Config: %d elements, %d beams, %d samples/ping\n", config.num_elements, config.num_beams, config.num_samples);
    printf("Channel data: %.1f MB, Beam output: %.1f MB, Delay table: %.1f KB\n\n",
        channel_bytes / 1e6f, beam_bytes / 1e6f, delay_bytes / 1024.0f);

    // --- Allocate host memory ---
    float* h_channels = (float*)malloc(channel_bytes);
    float* h_beams = (float*)malloc(beam_bytes);
    float* h_delays = (float*)malloc(delay_bytes);

    // Generate synthetic data (target at 30°, 15m range)
    printf("Generating synthetic multibeam data (target: 30°, 15m)...\n");
    generate_multibeam_data(h_channels, config, 42);
    build_delay_table(h_delays, config);

    // --- Allocate device memory ---
    float *d_channels, *d_beams, *d_delays;
    cudaMalloc(&d_channels, channel_bytes);
    cudaMalloc(&d_beams, beam_bytes);
    cudaMalloc(&d_delays, delay_bytes);

    cudaMemcpy(d_channels, h_channels, channel_bytes, cudaMemcpyHostToDevice);
    cudaMemcpy(d_delays, h_delays, delay_bytes, cudaMemcpyHostToDevice);

    cudaStream_t stream;
    cudaStreamCreate(&stream);

    // --- Warmup ---
    float warmup_ms;
    run_beamformer(d_channels, d_beams, d_delays, config, stream, warmup_ms);

    // --- Benchmark: 100 pings ---
    const int NUM_PINGS = 100;
    cudaEvent_t b_start, b_stop;
    cudaEventCreate(&b_start);
    cudaEventCreate(&b_stop);

    cudaEventRecord(b_start, stream);
    for (int p = 0; p < NUM_PINGS; p++) {
        float ms;
        run_beamformer(d_channels, d_beams, d_delays, config, stream, ms);
    }
    cudaEventRecord(b_stop, stream);
    cudaStreamSynchronize(stream);

    float total_ms;
    cudaEventElapsedTime(&total_ms, b_start, b_stop);
    cudaEventDestroy(b_start);
    cudaEventDestroy(b_stop);

    float avg_ms = total_ms / NUM_PINGS;
    float pings_per_sec = NUM_PINGS / (total_ms / 1000.0f);

    printf("\n--- Beamforming Performance ---\n");
    printf("  %d pings in %.2f ms\n", NUM_PINGS, total_ms);
    printf("  Average per ping: %.3f ms\n", avg_ms);
    printf("  Throughput: %.1f pings/sec (%.1f beams/sec)\n",
        pings_per_sec, pings_per_sec * config.num_beams);

    // --- Validate: find peak beam ---
    cudaMemcpy(h_beams, d_beams, beam_bytes, cudaMemcpyDeviceToHost);

    // Find beam with maximum energy (RMS)
    int peak_beam = -1;
    float peak_energy = -1.0f;

    for (int b = 0; b < config.num_beams; b++) {
        float energy = 0.0f;
        for (int s = 0; s < config.num_samples; s++) {
            float v = h_beams[b * config.num_samples + s];
            energy += v * v;
        }
        energy = sqrtf(energy / config.num_samples);  // RMS

        if (energy > peak_energy) {
            peak_energy = energy;
            peak_beam = b;
        }
    }

    // Convert beam index to angle
    float detected_angle = -config.beam_width_deg / 2.0f
                         + (float)peak_beam * config.beam_width_deg / (float)(config.num_beams - 1);

    printf("\n--- Beam Peak Detection ---\n");
    printf("  Target angle: 30.0°\n");
    printf("  Detected peak beam: %d → angle: %.1f° (error: %.1f°)\n",
        peak_beam, detected_angle, fabsf(detected_angle - 30.0f));
    printf("  Peak RMS energy: %.4f\n", peak_energy);

    // Show beam energy profile around peak
    printf("\n  Beam energy profile (around peak):\n");
    for (int b = peak_beam - 5; b <= peak_beam + 5; b++) {
        if (b < 0 || b >= config.num_beams) continue;
        float e = 0.0f;
        for (int s = 0; s < config.num_samples; s++) {
            float v = h_beams[b * config.num_samples + s];
            e += v * v;
        }
        e = sqrtf(e / config.num_samples);
        float a = -config.beam_width_deg / 2.0f + (float)b * config.beam_width_deg / (float)(config.num_beams - 1);
        printf("    beam %3d (%6.1f°): RMS %.4f %s\n", b, a, e, b == peak_beam ? " ← PEAK" : "");
    }

    // --- Vary array size benchmark ---
    printf("\n--- Array Size Scaling ---\n");
    int element_counts[] = {8, 16, 32, 64};
    for (int ei = 0; ei < 4; ei++) {
        int ne = element_counts[ei];
        BeamformerConfig cfg = config;
        cfg.num_elements = ne;

        float* d_ch2, *d_bm2, *d_dl2;
        size_t ch2 = (size_t)ne * cfg.num_samples * sizeof(float);
        size_t dl2 = (size_t)cfg.num_beams * ne * sizeof(float);
        size_t bm2 = (size_t)cfg.num_beams * cfg.num_samples * sizeof(float);
        cudaMalloc(&d_ch2, ch2); cudaMalloc(&d_bm2, bm2); cudaMalloc(&d_dl2, dl2);

        float* h_dl2 = (float*)malloc(dl2);
        cfg.element_spacing = 0.02f;
        build_delay_table(h_dl2, cfg);
        cudaMemcpy(d_dl2, h_dl2, dl2, cudaMemcpyHostToDevice);
        free(h_dl2);

        // Generate data for this array size
        float* h_ch2 = (float*)malloc(ch2);
        cfg.num_elements = ne;
        generate_multibeam_data(h_ch2, cfg, 42);
        cudaMemcpy(d_ch2, h_ch2, ch2, cudaMemcpyHostToDevice);
        free(h_ch2);

        // Benchmark 50 pings
        cudaEvent_t s1, s2;
        cudaEventCreate(&s1); cudaEventCreate(&s2);
        cudaEventRecord(s1, stream);
        for (int p = 0; p < 50; p++) {
            float ms;
            run_beamformer(d_ch2, d_bm2, d_dl2, cfg, stream, ms);
        }
        cudaEventRecord(s2, stream);
        cudaStreamSynchronize(stream);
        float t50;
        cudaEventElapsedTime(&t50, s1, s2);
        cudaEventDestroy(s1); cudaEventDestroy(s2);

        printf("  %2d elements: %.3f ms/ping, %.0f beams/sec\n",
            ne, t50 / 50.0f, 50.0f / (t50 / 1000.0f) * cfg.num_beams);

        cudaFree(d_ch2); cudaFree(d_bm2); cudaFree(d_dl2);
    }

    printf("\n=== Benchmark Complete ===\n");

    // Cleanup
    cudaStreamDestroy(stream);
    cudaFree(d_channels); cudaFree(d_beams); cudaFree(d_delays);
    free(h_channels); free(h_beams); free(h_delays);

    return 0;
}

// Duplicate delay table builder (linker needs it in this TU since beamformer declares it static)
static void build_delay_table(float* h_delays, const BeamformerConfig& config) {
    float sample_period = 1.0f / config.sample_rate;
    for (int b = 0; b < config.num_beams; b++) {
        float angle_deg = -config.beam_width_deg / 2.0f
                         + (float)b * config.beam_width_deg / (float)(config.num_beams - 1);
        float angle_rad = angle_deg * (float)M_PI / 180.0f;
        for (int e = 0; e < config.num_elements; e++) {
            float offset = (float)(e - config.num_elements / 2) * config.element_spacing;
            float delay_seconds = offset * sinf(angle_rad) / config.sound_velocity;
            h_delays[b * config.num_elements + e] = delay_seconds / sample_period;
        }
    }
}
