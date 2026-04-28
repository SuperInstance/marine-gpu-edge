# CLAUDE.md — Marine GPU Edge Computing

> You are Claude Code working for Forgemaster ⚒️ on a marine edge GPU computing project.
> This repo builds novel GPU technology for heterogeneous workstation↔edge distributed computing,
> with marine sensor fusion as the primary application domain.

## Project Overview

We're building a **distributed GPU compute mesh** that splits work between:
- **Workstation (eileen)**: RTX 4050 Ada Lovelace (SM 8.9, ~7.5GB VRAM) — training, heavy compute
- **Edge (jetsonclaw1)**: Jetson Orin Nano 8GB (SM 8.7, ARM64) — real-time inference, sensor fusion

The novel angle: **constraint-aware scheduling** that routes GPU kernels based on thermal envelope,
power budget, memory availability, precision requirements, and latency — using constraint theory
principles for real-time optimization.

## Repository Structure

```
marine-gpu-edge/
├── src/
│   ├── marine_sensor_fusion.cu   # CUDA kernels (NMEA, Kalman, sonar, nav constraints)
│   ├── mep_bridge.cpp            # Marine Edge Protocol + scheduler + server/client
│   ├── mep_bridge.h              # Header for MEP types and classes
│   ├── adaptive_precision.cu     # Precision switching controller
│   ├── fusion_pipeline.cu        # Multi-stage fusion pipeline orchestration
│   └── benchmarks/
│       ├── bench_nmea.cu         # NMEA parse throughput benchmark
│       ├── bench_kalman.cu       # Kalman filter throughput benchmark
│       └── bench_sonar.cu        # Sonar waterfall throughput benchmark
├── include/
│   └── marine_types.h            # Shared types for all kernels
├── docs/
│   ├── ARCHITECTURE.md           # Full architecture document
│   └── API.md                    # API reference
├── CMakeLists.txt                # Cross-compilation build system
├── CLAUDE.md                     # This file
└── README.md                     # Project README
```

## Build Requirements

### CUDA Compilation
- **CUDA Toolkit 12.6** available at `/usr/local/cuda-12.6`
- **nvcc** compiles and links successfully (verified)
- Target architectures: `sm_87` (Jetson Orin), `sm_89` (RTX 4050 Ada)
- For cross-compilation to Jetson (ARM64): use `aarch64-linux-gnu-g++` as host compiler

### Build Commands
```bash
# Workstation build (x86_64, sm_89)
nvcc -arch=sm_89 -O3 --use_fast_math -o marine_fusion marine_sensor_fusion.cu

# Cross-compile for Jetson (aarch64, sm_87)  
nvcc -arch=sm_87 --compiler-bindir=/usr/bin/aarch64-linux-gnu-g++ \
     -target aarch64-linux -O3 --use_fast_math -o marine_fusion_j marine_sensor_fusion.cu

# CMake build (preferred)
mkdir build && cd build
cmake .. -DCMAKE_BUILD_TYPE=Release
make -j$(nproc)
```

### Environment Constraints
- **rustc 1.75.0** on system — pin uuid≤1.4.1 if Rust code is used
- No OPENAI_API_KEY, no GROQ_API_KEY — no LLM API calls
- Max 2 concurrent `cargo check/build` — serialize Rust builds
- WSL2 environment — `/mnt/c/` gives Windows host access

## Architecture — What's Novel

### 1. Fused GPU Sensor Pipeline
Traditional: CPU parses NMEA → CPU updates Kalman → GPU does sonar.
Our approach: **Everything on GPU.** NMEA checksum+parse in parallel threads, warp-level
timestamp alignment, fused Kalman predict+update, sonar waterfall with TVG — single pipeline,
zero CPU-GPU round trips.

### 2. Adaptive Precision Controller
The GPU switches between FP32/FP16/TF32 at runtime based on:
- Thermal headroom (Jetson thermal throttle at ~85°C)
- Power budget (Orin Nano limited to ~15W)
- Position accuracy requirements (maritime safety)
- Memory pressure (shared covariance in FP16, critical values in FP32)

### 3. Marine Edge Protocol (MEP)
12-byte binary headers. Designed for edge latency. Message types for:
- GPU kernel offload + PTX deployment
- Zero-copy GPU memory hints
- Sensor batch streaming
- Navigation constraint violation alerts
- Constraint-aware load reporting

### 4. Constraint-Aware Scheduler
Scores each compute node on: thermal headroom, memory fit, current load, latency,
precision capability, deadline proximity. Uses multi-objective constraint optimization.
Jetson gets FP16 work (excellent tensor core throughput), workstation gets FP32/TF32.

### 5. GPU Navigation Constraint Propagation
Each thread checks one navigation constraint (depth > minimum, speed < limit, etc.)
in parallel using shared memory for conflict detection. Returns worst violation with
severity classification. This is constraint theory applied to maritime safety.

## Key Data Structures

### NavState (compact for edge)
- Position: FP16 (lat, lon)
- Velocity: FP16 (north, east in m/s)
- Heading: FP16 (radians)
- Heading rate: FP16 (rad/s)
- Depth: FP32 (meters — needs precision from sonar)
- Timestamp: FP32 (seconds)

### MEPHeader (12 bytes, packed)
- Magic: 0x4D524E45 ("MRNE")
- Version, Type, Length, Sequence

### MEPDeviceInfo (exchanged on connect)
- Hostname, GPU name, SM version, memory stats
- Temperature, power usage, power limit
- is_jetson flag for thermal-aware scheduling

## Testing Strategy

1. **Unit tests**: Each kernel in isolation with known inputs/outputs
2. **Benchmark suite**: Throughput in sentences/sec, pings/sec, constraint checks/sec
3. **Accuracy tests**: Kalman position error vs CPU reference implementation
4. **Stress tests**: Max throughput before thermal throttle on Jetson
5. **Integration test**: Full pipeline with synthetic NMEA + sonar data

## Coding Style

- C++17 / CUDA 12.6 features OK
- `__restrict__` on all kernel pointer params
- `extern "C"` for kernel host wrappers (C linkage for interop)
- `__device__ __host__` for utility functions shared between GPU/CPU
- Packed structs for network protocol (`__attribute__((packed))`)
- Minimal comments in hot paths, thorough comments on novel algorithms
- Error checking via CUDA macros: `CUDA_CHECK(call)` pattern
- No exceptions in GPU code — return error codes

## What To Build Next

Priority order:
1. **`include/marine_types.h`** — Extract shared types from the .cu files
2. **`CMakeLists.txt`** — Proper build system with cross-compilation targets
3. **`marine_sensor_fusion.cu`** — Add CUDA_CHECK macros, fix the parse kernel (needs proper packed sentence layout), add unit tests
4. **`benchmarks/bench_nmea.cu`** — NMEA parse benchmark (compare GPU vs CPU)
5. **`benchmarks/bench_kalman.cu`** — Kalman throughput benchmark
6. **`adaptive_precision.cu`** — Implement the runtime precision controller
7. **`fusion_pipeline.cu`** — Multi-stage pipeline with stream synchronization
8. **`README.md`** — Ship-ready project README with build instructions
9. **Test compilation** — Verify both x86_64 and aarch64 targets compile

## What NOT To Do

- Don't use MPI, gRPC, or heavy frameworks — MEP is deliberately lightweight
- Don't use Eigen or other CPU math libs — CUDA native math only
- Don't link against libcuda directly — use CUDA runtime API
- Don't write Python wrappers — this is C/CUDA for edge deployment
- Don't add LLM dependencies — no API keys available
- Don't use Rust for this project — keep it pure C/CUDA for Jetson compatibility

## Git

- Push to `forgemaster` remote: `git push forgemaster master`
- Commit messages: `marine-gpu: brief description of change`
- Work in `/tmp/marine-gpu-edge/` initially, then move to proper repo
