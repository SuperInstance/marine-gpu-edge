/**
 * monte_carlo_uncertainty.cu — GPU-accelerated Monte Carlo uncertainty propagation
 *
 * Novel: Parallel sampling of particle ensembles for navigation uncertainty quantification.
 * Applies constraint theory by tracking confidence bounds across simulation.
 *
 * For marine navigation: position uncertainty grows over time due to:
 * - GPS error (multipath, ionosphere, satellite geometry)
 * - Dead reckoning error (current drift, compass bias)
 * - Sensor noise (sonar depth, speed log)
 *
 * Monte Carlo: sample 10K-100K perturbations from initial covariance,
 * propagate each through dynamics, compute final covariance statistics.
 *
 * Author: Forgemaster ⚒️
 * License: MIT
 */

#include "marine_types.h"
#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <curand_kernel.h>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>

// ============================================================
// GPU Kernel: Monte Carlo particle propagation
// ============================================================

/**
 * Each thread = one particle.
 * Shared memory: per-thread PRNG state (curandStatePhilox4_32)
 *
 * Dynamics: simplified constant-velocity model with process noise
 * x[t+1] = x[t] + v[t] × dt + process_noise
 * v[t+1] = v[t] + velocity_noise
 *
 * Process noise models: GPS position error (per step), GPS velocity error,
 * current drift (random walk in velocity).
 */
__global__ void mc_propagate_particles(
    float* __restrict__ particle_pos,  // [num_particles × 2] meters offset (lat, lon)
    float* __restrict__ particle_vel,  // [num_particles × 2] m/s (vn, ve)
    const MonteCarloConfig config,
    unsigned long long seed
) {
    int pidx = blockIdx.x * blockDim.x + threadIdx.x;
    if (pidx >= config.num_particles) return;

    // Initialize PRNG for this thread
    curandStatePhilox4_32_10 state;
    curand_init(seed, pidx, 0, &state);

    // Position: offset from initial position (meters north, meters east)
    // Velocity: north and east components (m/s)
    float pos_x = 0.0f, pos_y = 0.0f;
    float vel_x = 0.0f, vel_y = 0.0f;

    // Apply initial GPS position error (Gaussian, 2m sigma)
    float4 gps_err = curand_normal4(&state);
    pos_x += gps_err.x * config.gps_pos_error;
    pos_y += gps_err.y * config.gps_pos_error;

    // Apply initial GPS velocity error (Gaussian, 0.3 m/s sigma)
    float4 vel_err = curand_normal4(&state);
    vel_x += vel_err.x * config.gps_vel_error;
    vel_y += vel_err.y * config.gps_vel_error;

    // Propagate through time steps
    for (int step = 0; step < config.num_steps; step++) {
        // Dead reckoning: position += velocity × dt
        pos_x += vel_x * config.dt;
        pos_y += vel_y * config.dt;

        // Apply per-step GPS position correction (simulates periodic fixes)
        // Every 10 steps, apply GPS error again
        if (step % 10 == 0) {
            float4 gp = curand_normal4(&state);
            // GPS error reduces as particle converges (simple model: 2× initial → 0.5× at end)
            float gps_scale = config.gps_pos_error * (1.0f - 0.5f * (float)step / (float)config.num_steps);
            pos_x += gp.x * gps_scale;
            pos_y += gp.y * gps_scale;
        }

        // Current drift: random walk in velocity
        float4 drift = curand_normal4(&state);
        vel_x += drift.x * config.current_drift * config.dt;
        vel_y += drift.y * config.current_drift * config.dt;
    }

    // Write final particle state
    particle_pos[pidx * 2 + 0] = pos_x;
    particle_pos[pidx * 2 + 1] = pos_y;
    particle_vel[pidx * 2 + 0] = vel_x;
    particle_vel[pidx * 2 + 1] = vel_y;
}

// ============================================================
// GPU Kernel: Compute ensemble statistics (mean + std)
// ============================================================

/**
 * One block for lat statistics, one for lon.
 * Warp-level reduction for mean, then parallel for variance.
 */
__global__ void mc_compute_stats(
    const float* __restrict__ particle_pos,
    float* __restrict__ d_mean,   // [2] output (lat_mean, lon_mean)
    float* __restrict__ d_std,    // [2] output (lat_std, lon_std)
    int num_particles
) {
    int idx = threadIdx.x;
    int axis = blockIdx.x;  // 0 = lat (x), 1 = lon (y)
    if (axis >= 2) return;

    // Load this particle's position for this axis
    float val = particle_pos[idx * 2 + axis];

    // Warp-level reduction for mean
    for (int offset = 16; offset > 0; offset /= 2) {
        val += __shfl_down_sync(0xFFFFFFFF, val, offset);
    }

    // Lane 0 writes mean to shared memory
    __shared__ float sm_mean;
    if (idx == 0) sm_mean = val / (float)num_particles;
    __syncthreads();

    float mean = sm_mean;

    // Variance: sum of squared differences
    val = (particle_pos[idx * 2 + axis] - mean) * (particle_pos[idx * 2 + axis] - mean);

    for (int offset = 16; offset > 0; offset /= 2) {
        val += __shfl_down_sync(0xFFFFFFFF, val, offset);
    }

    // Lane 0: compute std from variance and write
    if (idx == 0) {
        float variance = val / (float)num_particles;
        d_mean[axis] = mean;
        d_std[axis] = sqrtf(variance);
    }
}

// ============================================================
// Host: Run Monte Carlo uncertainty propagation
// ============================================================

extern "C" cudaError_t run_monte_carlo(
    const MonteCarloConfig& config,
    float& out_mean_lat,
    float& out_mean_lon,
    float& out_std_lat,
    float& out_std_lon,
    cudaStream_t stream,
    float& out_time_ms
) {
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    size_t pos_bytes = (size_t)config.num_particles * 2 * sizeof(float);
    size_t vel_bytes = (size_t)config.num_particles * 2 * sizeof(float);
    size_t stats_bytes = 2 * sizeof(float);

    // Allocate device memory
    float* d_particle_pos = nullptr;
    float* d_particle_vel = nullptr;
    float* d_mean = nullptr;
    float* d_std = nullptr;

    cudaMallocAsync(&d_particle_pos, pos_bytes, stream);
    cudaMallocAsync(&d_particle_vel, vel_bytes, stream);
    cudaMallocAsync(&d_mean, stats_bytes, stream);
    cudaMallocAsync(&d_std, stats_bytes, stream);

    // Launch particle propagation
    int threads = 256;
    int blocks = (config.num_particles + threads - 1) / threads;

    cudaEventRecord(start, stream);

    mc_propagate_particles<<<blocks, threads, 0, stream>>>(
        d_particle_pos, d_particle_vel, config,
        42ULL  // fixed seed for reproducibility
    );

    // Compute ensemble statistics (2 blocks: lat and lon)
    mc_compute_stats<<<2, 256, 0, stream>>>(
        d_particle_pos, d_mean, d_std, config.num_particles
    );

    cudaEventRecord(stop, stream);
    cudaStreamSynchronize(stream);

    float ms;
    cudaEventElapsedTime(&ms, start, stop);
    out_time_ms = ms;

    // Copy results back
    float h_mean[2], h_std[2];
    cudaMemcpyAsync(h_mean, d_mean, stats_bytes, cudaMemcpyDeviceToHost, stream);
    cudaMemcpyAsync(h_std, d_std, stats_bytes, cudaMemcpyDeviceToHost, stream);
    cudaStreamSynchronize(stream);

    out_mean_lat = h_mean[0];
    out_mean_lon = h_mean[1];
    out_std_lat = h_std[0];
    out_std_lon = h_std[1];

    // Cleanup
    cudaFreeAsync(d_particle_pos, stream);
    cudaFreeAsync(d_particle_vel, stream);
    cudaFreeAsync(d_mean, stream);
    cudaFreeAsync(d_std, stream);
    cudaEventDestroy(start);
    cudaEventDestroy(stop);

    return cudaGetLastError();
}
