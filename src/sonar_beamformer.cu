/**
 * sonar_beamformer.cu — GPU-accelerated delay-and-sum beamforming for multibeam sonar
 *
 * Novel: Warp-cooperative beamforming with shared memory delay tables,
 * enabling real-time multibeam processing on Jetson Orin Nano.
 *
 * Physics: For a linear array of N elements, beamforming steers a beam
 * at angle theta by applying time delays: delay(i) = i * d * sin(theta) / c,
 * where d = element spacing, c = sound velocity. The delay-and-sum beamformer
 * computes: beam(theta, t) = sum_i(signal(i, t - delay(i))).
 *
 * GPU strategy: One CUDA block per beam, one thread per array element.
 * Shared memory holds the precomputed delay table and beam accumulator.
 * Uses atomicAdd on shared memory float (supported SM 7.0+).
 *
 * Author: Forgemaster ⚒️
 * License: MIT
 */

#include "marine_types.h"
#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>

// ============================================================
// GPU Kernel: Delay-and-sum beamforming
// ============================================================

/**
 * Each block = one beam direction.
 * Each thread = one array element.
 * Shared memory: delay_table[num_elements] + beam_accum[num_samples]
 *
 * Strategy: Each thread zero-extends its channel data, applies fractional
 * delay via linear interpolation, and atomicAdds to the shared beam buffer.
 * Thread 0 normalizes and applies Hamming window, then writes to global.
 */
__global__ void beamform_delay_and_sum(
    const float* __restrict__ channel_data,  // [num_elements × num_samples]
    float* __restrict__ beam_output,          // [num_beams × num_samples]
    const float* __restrict__ delay_table,    // [num_beams × num_elements] in samples
    int num_elements,
    int num_samples,
    int num_beams
) {
    extern __shared__ char smem[];

    int beam_idx = blockIdx.x;
    if (beam_idx >= num_beams) return;

    int elem_idx = threadIdx.x;

    // Shared memory layout
    float* sm_delays = reinterpret_cast<float*>(smem);
    float* sm_beam   = reinterpret_cast<float*>(smem + num_elements * sizeof(float));

    // Load delay for this beam+element
    if (elem_idx < num_elements) {
        sm_delays[elem_idx] = delay_table[beam_idx * num_elements + elem_idx];
    }

    // Zero-initialize beam accumulator (cooperative across threads)
    for (int s = elem_idx; s < num_samples; s += blockDim.x) {
        sm_beam[s] = 0.0f;
    }
    __syncthreads();

    // Each element thread: apply delay and accumulate
    if (elem_idx < num_elements) {
        float delay_samples = sm_delays[elem_idx];
        int delay_int = (int)floorf(delay_samples);
        float delay_frac = delay_samples - (float)delay_int;

        const float* my_channel = channel_data + (size_t)elem_idx * num_samples;

        for (int s = 0; s < num_samples; s++) {
            int s_delayed = s + delay_int;
            float sample = 0.0f;

            if (s_delayed >= 0 && s_delayed < num_samples - 1) {
                sample = my_channel[s_delayed] * (1.0f - delay_frac)
                       + my_channel[s_delayed + 1] * delay_frac;
            } else if (s_delayed >= 0 && s_delayed < num_samples) {
                sample = my_channel[s_delayed];
            }

            atomicAdd(&sm_beam[s], sample);
        }
    }
    __syncthreads();

    // Thread 0: normalize and window, write to global
    if (elem_idx == 0) {
        float inv_n = 1.0f / (float)num_elements;
        float* out = beam_output + (size_t)beam_idx * num_samples;

        for (int s = 0; s < num_samples; s++) {
            float hamming = 0.54f - 0.46f * cosf(2.0f * 3.14159265358979f
                            * (float)s / (float)(num_samples - 1));
            out[s] = sm_beam[s] * inv_n * hamming;
        }
    }
}

// ============================================================
// Host: Build delay table
// ============================================================

static void build_delay_table(
    float* h_delays,
    const BeamformerConfig& config
) {
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

// ============================================================
// Host: Run beamformer
// ============================================================

extern "C" cudaError_t run_beamformer(
    const float* d_channel_data,
    float* d_beam_output,
    const float* d_delay_table,
    const BeamformerConfig& config,
    cudaStream_t stream,
    float& out_time_ms
) {
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    int smem_size = config.num_elements * sizeof(float) + config.num_samples * sizeof(float);

    cudaEventRecord(start, stream);

    beamform_delay_and_sum<<<config.num_beams, config.num_elements, smem_size, stream>>>(
        d_channel_data, d_beam_output, d_delay_table,
        config.num_elements, config.num_samples, config.num_beams
    );

    cudaEventRecord(stop, stream);
    cudaStreamSynchronize(stream);

    float ms;
    cudaEventElapsedTime(&ms, start, stop);
    out_time_ms = ms;

    cudaEventDestroy(start);
    cudaEventDestroy(stop);

    return cudaGetLastError();
}

// ============================================================
// Host: Generate synthetic multibeam sonar data
// ============================================================

extern "C" void generate_multibeam_data(
    float* h_data,
    const BeamformerConfig& config,
    int seed
) {
    srand(seed);

    for (int e = 0; e < config.num_elements; e++) {
        for (int s = 0; s < config.num_samples; s++) {
            float noise = ((float)rand() / (float)RAND_MAX - 0.5f) * 0.05f;

            // Simulate a target at 30 degrees, range 15m
            float target_angle = 30.0f * (float)M_PI / 180.0f;
            float target_range_samples = 15.0f * config.sample_rate / config.sound_velocity;
            float element_offset = (float)(e - config.num_elements / 2) * config.element_spacing;
            float target_delay = element_offset * sinf(target_angle) / config.sound_velocity
                               * config.sample_rate;

            float target_signal = 0.0f;
            float dist = fabsf((float)s - target_range_samples - target_delay);
            if (dist < 50.0f) {
                float freq = 37500.0f;
                float t_shifted = ((float)s - target_delay) / config.sample_rate;
                float phase = 2.0f * (float)M_PI * freq * t_shifted;
                target_signal = 1.0f * expf(-dist * dist / 400.0f) * sinf(phase);
            }

            h_data[e * config.num_samples + s] = noise + target_signal;
        }
    }
}
