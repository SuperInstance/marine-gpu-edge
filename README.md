# marine-gpu-edge

Distributed GPU compute mesh for marine sensor fusion.
Splits workloads between a workstation RTX 4050 and a Jetson Orin Nano edge device
using a constraint-aware scheduler and a lightweight binary protocol (MEP).

---

## Novel Contributions

| Contribution | Description |
|---|---|
| **Fused GPU sensor pipeline** | NMEA parse → Kalman predict/update → sonar waterfall in a single GPU pipeline. Zero CPU-GPU round trips. |
| **Adaptive precision controller** | Switches FP32/FP16/TF32 at runtime based on thermal headroom, power budget, and position accuracy requirements. |
| **Marine Edge Protocol (MEP)** | 16-byte binary header. Designed for LAN edge latency. Supports GPU kernel offload, PTX deployment, zero-copy hints, and constraint alerts. |
| **Constraint-aware scheduler** | Multi-objective scoring: thermal, memory, load, latency, precision capability, deadline proximity. Routes FP16 work to Jetson, FP32/TF32 to workstation. |
| **GPU navigation constraint propagation** | Each thread checks one safety constraint in parallel via shared memory. Returns worst violation with severity classification. |

---

## Hardware

| Node | GPU | CUDA SM | Memory | Role |
|---|---|---|---|---|
| `eileen` (workstation) | RTX 4050 Ada Lovelace | SM 8.9 | ~7.5 GB | Training, heavy compute, FP32/TF32 |
| `jetsonclaw1` (edge) | Jetson Orin Nano 8 GB | SM 8.7 | 8 GB unified | Real-time inference, sensor fusion, FP16 |

---

## Verification status

This repository was last hardened on the `production-round3-2026-07-10` branch.
The host used for that pass had **no CUDA toolkit, no `nvcc`, and no GPU**,
so only the genuinely CPU-testable portions could be verified.

| Claim | Status | Notes |
|---|---|---|
| MEP 16-byte binary header | ✅ Verified | `static_assert(sizeof(MEPHeader) == 16)` enforced at compile time; serialisation, validation, and send/recv framing covered by `tests/test_mep_bridge.cpp` |
| Constraint-aware scheduler logic | ✅ Verified | Thermal, memory, load, FP16, and PTX scoring tested in `tests/test_mep_bridge.cpp` |
| CUDA kernel behaviour (NMEA parse, Kalman, sonar waterfall, constraint check) | 🔮 Requires GPU | Needs `nvcc` + CUDA 12.6 + actual RTX 4050 / Jetson Orin hardware |
| GPU scheduling / runtime precision switching | 🔮 Requires GPU | Depends on kernel behaviour above |
| Benchmark numbers in the table below | 🔮 Requires GPU | Currently placeholders (`TBD`) until run on real hardware |

Run the CPU-only tests with plain g++ (no CUDA needed):

```bash
g++ -std=c++17 -I include -I src tests/test_mep_bridge.cpp src/mep_bridge.cpp -o tests/test_mep_bridge
./tests/test_mep_bridge
```

---

## Build

### Requirements

- CUDA Toolkit 12.6 at `/usr/local/cuda-12.6`
- CMake ≥ 3.20
- C++17-capable host compiler (GCC 11+ recommended)
- For Jetson cross-compile: `aarch64-linux-gnu-g++`

### Workstation build (x86_64, SM 8.9 + SM 8.7 fat binary)

```bash
mkdir build && cd build
cmake .. -DCMAKE_BUILD_TYPE=Release
make -j$(nproc)
```

### Jetson-only cross-compile (aarch64, SM 8.7)

```bash
mkdir build-jetson && cd build-jetson
cmake .. -DCMAKE_BUILD_TYPE=Release -DCROSS_COMPILE_JETSON=ON
make -j$(nproc)
```

### Manual nvcc (quick single-file test)

```bash
# Workstation
nvcc -arch=sm_89 -O3 --use_fast_math -I include \
     -o marine_fusion src/marine_sensor_fusion.cu

# Jetson cross-compile
nvcc -arch=sm_87 --compiler-bindir=/usr/bin/aarch64-linux-gnu-g++ \
     -O3 --use_fast_math -I include \
     -o marine_fusion_j src/marine_sensor_fusion.cu
```

### CMake targets

| Target | Description |
|---|---|
| `marine_fusion` | Static library — CUDA kernels |
| `mep_bridge_test` | MEP protocol + scheduler smoke test |
| `bench_nmea` | NMEA parse throughput benchmark |

---

## Architecture

```
┌─────────────────────────────────────────────────────┐
│  Sensor inputs (NMEA sentences, sonar pings)         │
└───────────────────────┬─────────────────────────────┘
                        │
          ┌─────────────▼────────────┐
          │  MEPBridge + Scheduler   │  (LAN, port 7847)
          │  Constraint-aware route  │
          └──────┬──────────┬────────┘
                 │          │
    ┌────────────▼──┐  ┌────▼──────────────┐
    │   eileen      │  │   jetsonclaw1      │
    │  RTX 4050     │  │  Jetson Orin Nano  │
    │  SM 8.9       │  │  SM 8.7            │
    │               │  │                    │
    │ parse_nmea    │  │ kalman_update       │
    │ kalman_predict│  │ sonar_waterfall     │
    │ (FP32/TF32)   │  │ check_constraints  │
    └───────────────┘  │ (FP16)             │
                       └────────────────────┘
```

### Kernel pipeline

1. **`parse_nmea_batch`** — one thread per sentence, stride-based addressing, checksum + field extraction
2. **`kalman_predict_batch`** — EKF prediction with FP16 covariance, FP32 dynamics
3. **`sonar_waterfall`** — warp-cooperative ping processing, TVG correction in shared memory
4. **`check_nav_constraints`** — one thread per constraint per state, parallel violation detection

---

## Benchmark Results

*Run on workstation (RTX 4050 Ada, SM 8.9, CUDA 12.6)*

| Benchmark | GPU kernel | GPU end-to-end | CPU | Speedup (kernel) |
|---|---|---|---|---|
| NMEA parse (100K sentences) | TBD | TBD | TBD | TBD |
| Kalman predict (100K states) | TBD | TBD | TBD | TBD |
| Sonar waterfall (1K pings) | TBD | TBD | TBD | TBD |

Run benchmarks:
```bash
cd build
./bench_nmea 100000
```

---

## Project Structure

```
marine-gpu-edge/
├── include/
│   └── marine_types.h          # All shared types + CUDA_CHECK macros
├── src/
│   ├── marine_sensor_fusion.cu # CUDA kernels + host wrappers
│   ├── mep_bridge.h            # MEPBridge + ConstraintScheduler declarations
│   └── mep_bridge.cpp          # MEP protocol implementation
├── benchmarks/
│   └── bench_nmea.cu           # NMEA parse throughput benchmark
├── tests/
│   └── test_mep_bridge.cpp     # CPU-only unit tests for MEP bridge + scheduler
├── .github/workflows/
│   └── mep_bridge.yml          # CI for the CPU-only bridge tests
├── docs/
│   ├── ARCHITECTURE.md
│   └── API.md
├── CMakeLists.txt
└── README.md
```

---

## License

MIT — see source file headers.
