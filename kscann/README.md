# KScaNN – Simplified IVF+PQ Approximate Nearest Neighbour Search

This is a modular C++ implementation based on the algorithm described in  
"KScaNN: Scalable Approximate Nearest Neighbor Search on Kunpeng" (arXiv:2511.03298).

## Features
- Inverted File (IVF) index with k‑means clustering
- Product Quantisation (PQ) for compressed storage
- SIMD‑accelerated distance computation (ARM NEON support)
- Re‑ranking for improved accuracy

## Build
```bash
mkdir build && cd build
cmake .. -DKSCANN_USE_NEON=OFF    # enable NEON on ARM64
make -j4

# Baseline
./bench_glove glove-100_100k.fvecs

# RAIRS optimised
./bench_glove glove-100_100k.fvecs rairs
