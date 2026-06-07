#pragma once

#include <cstdint>

#ifdef __AVX2__
#include <immintrin.h>
#endif

namespace kscann {

// Compute approximate distances for a block of 32 points using AVX2.
// lut: flat [subspaces][256] uint8_t distances
// codes: flat array of PQ codes for 32 points: layout [point][subspace]
// out_dist: 32 floats (output)
void compute_distances_block_avx2(const uint8_t* lut, const uint8_t* codes,
                                  float* out_dist, int num_subspaces);

} // namespace kscann