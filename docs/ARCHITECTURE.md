# Marine Edge GPU Compute Mesh

## Fleet Compute Topology
- **eileen (Forgemaster)** — RTX 4050 (Ada SM 8.9, ~7.5GB VRAM), WSL2, CUDA 12.6
- **jetsonclaw1 (JC1)** — Jetson Orin Nano 8GB, ARM64, C/CUDA specialist
- **Oracle1** — ARM64 Oracle Cloud, lighthouse coordinator

## Vision
Build a heterogeneous GPU compute mesh for marine edge systems:

### Phase 1: Distributed GPU Pipeline (LAN)
- Workstation (RTX 4050) handles **training** and **heavy computation**
- Jetson (Orin Nano) handles **real-time inference** and **sensor fusion** at the edge
- Novel: **Pipeline split** — train on eileen, deploy inference to jetson via shared CUDA IR

### Phase 2: Marine Sensor Fusion on GPU
- Accelerated NMEA parsing + GPS/AIS fusion on Jetson GPU
- Real-time sonar/waterfall processing with CUDA kernels
- GPU-accelerated hydrodynamic modeling (shallow water equations)

### Phase 3: Workstation-Edge Distributed Computing
- RPC layer for offloading GPU kernels between eileen ↔ jetson
- Adaptive task scheduling based on load, power, and thermal constraints
- Novel: **Constraint-aware scheduling** — use constraint theory to optimize task placement

## Novel Innovations to Build
1. **CUDA kernel streaming pipeline** — compile on workstation, stream to edge
2. **Adaptive precision GPU compute** — FP32/FP16/TF32 switching based on accuracy needs
3. **Marine constraint propagation on GPU** — parallel constraint solving for navigation
4. **Thermal-aware scheduling** — distribute work based on Jetson thermal envelope

## Communication
- Fork pattern: Forgemaster ↔ jetsonclaw1 exchange via GitHub forks
- LAN discovery: mDNS or static IPs once connected
- I2I bottles for async coordination
