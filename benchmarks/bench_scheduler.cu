/**
 * bench_scheduler.cu — Constraint-aware task scheduler benchmark
 *
 * Tests the GPU-parallel multi-objective scheduler with synthetic workloads.
 * Scenarios: normal, deadline-stress, thermal-constrained, memory-overflow.
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

// From adaptive_precision.cu
extern "C" int schedule_tasks_greedy(
    const ComputeTask* h_tasks,
    const NodeProfile* h_nodes,
    int num_tasks,
    int num_nodes,
    int* assignment,
    cudaStream_t stream);

// ============================================================
// Helpers
// ============================================================

static void print_assignment(const int* assignment, const NodeProfile* nodes,
                              int num_tasks, int num_nodes, const char* label) {
    int* per_node = (int*)calloc(num_nodes, sizeof(int));
    for (int t = 0; t < num_tasks; t++) {
        if (assignment[t] >= 0 && assignment[t] < num_nodes)
            per_node[assignment[t]]++;
    }

    printf("  [%s] Assignment:", label);
    for (int n = 0; n < num_nodes; n++) {
        const char* tag = nodes[n].is_jetson ? " (Jetson)" : " (Desktop)";
        printf("  node-%d%s: %d tasks", n, tag, per_node[n]);
    }
    int unassigned = 0;
    for (int t = 0; t < num_tasks; t++) if (assignment[t] < 0) unassigned++;
    if (unassigned > 0) printf("  unassigned: %d", unassigned);
    printf("\n");

    free(per_node);
}

// ============================================================
// Scenario generators
// ============================================================

// Normal mixed workload
static void scenario_normal(ComputeTask* tasks, int num_tasks) {
    for (int i = 0; i < num_tasks; i++) {
        tasks[i].compute_gflops = 0.5f + ((float)(i % 10) * 0.3f);
        tasks[i].memory_bytes = (size_t)(1 + (i % 8)) * 128 * 1024;  // 128KB-1MB
        tasks[i].deadline_ms = 5.0f + (float)(i % 20);
        tasks[i].precision_req = (i % 3 == 0) ? PREC_FP32 :
                                 (i % 3 == 1) ? PREC_FP16 : PREC_TF32;
        tasks[i].accuracy_target = 0.5f + (float)(i % 10) * 0.1f;
        tasks[i].priority = 0.3f + (float)(i % 7) * 0.1f;
    }
}

// High-deadline stress: tight deadlines, should cause unassignments
static void scenario_deadline_stress(ComputeTask* tasks, int num_tasks) {
    for (int i = 0; i < num_tasks; i++) {
        tasks[i].compute_gflops = 5.0f + (float)i * 0.1f;  // heavy tasks
        tasks[i].memory_bytes = (size_t)(1 + (i % 4)) * 256 * 1024;
        tasks[i].deadline_ms = 0.5f;  // very tight — should fail most
        tasks[i].precision_req = PREC_FP32;
        tasks[i].accuracy_target = 0.1f;
        tasks[i].priority = 1.0f;
    }
}

// Thermal constrained: nodes are hot, should force FP16/low-power assignments
static void scenario_thermal_stress(ComputeTask* tasks, int num_tasks) {
    for (int i = 0; i < num_tasks; i++) {
        tasks[i].compute_gflops = 2.0f;
        tasks[i].memory_bytes = 512 * 1024;
        tasks[i].deadline_ms = 10.0f;
        tasks[i].precision_req = (i % 2 == 0) ? PREC_FP16 : PREC_FP32;
        tasks[i].accuracy_target = 1.0f;
        tasks[i].priority = 0.5f;
    }
}

// Memory pressure: tasks need more memory than available
static void scenario_memory_pressure(ComputeTask* tasks, int num_tasks) {
    for (int i = 0; i < num_tasks; i++) {
        tasks[i].compute_gflops = 0.5f;
        tasks[i].memory_bytes = (size_t)(512 + i * 64) * 1024 * 1024;  // 512MB+
        tasks[i].deadline_ms = 100.0f;
        tasks[i].precision_req = PREC_FP16;
        tasks[i].accuracy_target = 5.0f;
        tasks[i].priority = 0.8f;
    }
}

// ============================================================
// Main
// ============================================================

int main() {
    printf("=== bench_scheduler: Constraint-Aware Task Scheduler ===\n\n");

    cudaDeviceProp prop;
    cudaGetDeviceProperties(&prop, 0);
    printf("GPU: %s (SM %d.%d)\n\n", prop.name, prop.major, prop.minor);

    cudaStream_t stream;
    cudaStreamCreate(&stream);

    // --- Node profiles ---
    // Node 0: Workstation (RTX 4050)
    NodeProfile nodes[2];
    memset(&nodes[0], 0, sizeof(NodeProfile));
    nodes[0].node_id = 0;
    nodes[0].peak_gflops = 8.9f;     // ~8.9 TFLOPS FP32
    nodes[0].memory_total = (size_t)8 * 1024 * 1024 * 1024;
    nodes[0].memory_used  = (size_t)2 * 1024 * 1024 * 1024;  // 2GB used
    nodes[0].temp_c = 45.0f;
    nodes[0].temp_limit_c = 87.0f;
    nodes[0].power_w = 35.0f;
    nodes[0].power_limit_w = 115.0f;
    nodes[0].network_latency_ms = 0.1f;
    nodes[0].precision_mask = (1 << PREC_FP32) | (1 << PREC_FP16) | (1 << PREC_TF32);
    nodes[0].is_jetson = 0;

    // Node 1: Jetson Orin Nano
    memset(&nodes[1], 0, sizeof(NodeProfile));
    nodes[1].node_id = 1;
    nodes[1].peak_gflops = 2.0f;     // ~2 TFLOPS FP32
    nodes[1].memory_total = (size_t)8 * 1024 * 1024 * 1024;
    nodes[1].memory_used  = (size_t)3 * 1024 * 1024 * 1024;  // 3GB used
    nodes[1].temp_c = 52.0f;
    nodes[1].temp_limit_c = 85.0f;
    nodes[1].power_w = 10.0f;
    nodes[1].power_limit_w = 15.0f;
    nodes[1].network_latency_ms = 1.0f;
    nodes[1].precision_mask = (1 << PREC_FP32) | (1 << PREC_FP16) | (1 << PREC_TF32);
    nodes[1].is_jetson = 1;

    const int num_nodes = 2;
    const int task_counts[] = {100, 500, 1000, 5000};
    const int num_scenarios = sizeof(task_counts) / sizeof(task_counts[0]);

    for (int s = 0; s < num_scenarios; s++) {
        int N = task_counts[s];

        printf("--- Scenario %d: %d tasks, normal workload ---\n", s+1, N);
        ComputeTask* tasks = (ComputeTask*)malloc(N * sizeof(ComputeTask));
        int* assignment = (int*)malloc(N * sizeof(int));

        scenario_normal(tasks, N);

        // Benchmark
        cudaEvent_t s0, s1;
        cudaEventCreate(&s0);
        cudaEventCreate(&s1);
        cudaEventRecord(s0, stream);

        int rc = schedule_tasks_greedy(tasks, nodes, N, num_nodes, assignment, stream);

        cudaEventRecord(s1, stream);
        cudaStreamSynchronize(stream);
        float ms;
        cudaEventElapsedTime(&ms, s0, s1);
        cudaEventDestroy(s0);
        cudaEventDestroy(s1);

        if (rc != 0) {
            printf("  Scheduler failed!\n");
        } else {
            printf("  Schedule time: %.3f ms\n", ms);
            print_assignment(assignment, nodes, N, num_nodes, "normal");

            // Precision distribution
            int fp32_count = 0, fp16_count = 0, tf32_count = 0;
            for (int t = 0; t < N; t++) {
                if (assignment[t] == 1 && tasks[t].precision_req == PREC_FP16)
                    fp16_count++;
                if (assignment[t] == 0 && tasks[t].precision_req == PREC_FP32)
                    fp32_count++;
                if (tasks[t].precision_req == PREC_TF32)
                    tf32_count++;
            }
            printf("  FP16→Jetson: %d, FP32→Desktop: %d, TF32 total: %d\n",
                fp16_count, fp32_count, tf32_count);
        }

        free(tasks);
        free(assignment);
        printf("\n");
    }

    // --- Edge case scenarios (100 tasks each) ---
    const int EDGE_N = 100;
    ComputeTask* edge_tasks = (ComputeTask*)malloc(EDGE_N * sizeof(ComputeTask));
    int* edge_assign = (int*)malloc(EDGE_N * sizeof(int));

    // Deadline stress
    printf("--- Edge Case: Deadline Stress (100 tasks, 0.5ms deadline) ---\n");
    scenario_deadline_stress(edge_tasks, EDGE_N);
    schedule_tasks_greedy(edge_tasks, nodes, EDGE_N, num_nodes, edge_assign, stream);
    print_assignment(edge_assign, nodes, EDGE_N, num_nodes, "deadline");
    printf("\n");

    // Thermal stress — make nodes hot
    printf("--- Edge Case: Thermal Stress (both nodes at 84°C) ---\n");
    scenario_thermal_stress(edge_tasks, EDGE_N);
    nodes[0].temp_c = 84.0f;
    nodes[1].temp_c = 84.0f;
    schedule_tasks_greedy(edge_tasks, nodes, EDGE_N, num_nodes, edge_assign, stream);
    print_assignment(edge_assign, nodes, EDGE_N, num_nodes, "thermal");
    nodes[0].temp_c = 45.0f;
    nodes[1].temp_c = 52.0f;
    printf("\n");

    // Memory pressure
    printf("--- Edge Case: Memory Pressure (512MB+ per task) ---\n");
    scenario_memory_pressure(edge_tasks, EDGE_N);
    schedule_tasks_greedy(edge_tasks, nodes, EDGE_N, num_nodes, edge_assign, stream);
    print_assignment(edge_assign, nodes, EDGE_N, num_nodes, "memory");
    printf("\n");

    // Power budget exhaustion
    printf("--- Edge Case: Jetson power budget (5W remaining) ---\n");
    scenario_normal(edge_tasks, EDGE_N);
    nodes[1].power_w = 10.0f;
    nodes[1].power_limit_w = 10.0f;  // already at limit
    schedule_tasks_greedy(edge_tasks, nodes, EDGE_N, num_nodes, edge_assign, stream);
    print_assignment(edge_assign, nodes, EDGE_N, num_nodes, "power");
    nodes[1].power_limit_w = 15.0f;
    printf("\n");

    free(edge_tasks);
    free(edge_assign);
    cudaStreamDestroy(stream);

    printf("=== Benchmark Complete ===\n");
    return 0;
}
