#pragma once

#include <cstdint>
#include <vector>

namespace kscann {

// Compute approximate distances for a block of points (size = 32).
// Input:
//   lut: flat array of 8‑bit distances, layout [subspace][centroid]
//   codes: pointer to PQ codes for the block, layout [point][subspace] (packed)
//   num_subspaces: number of subspaces (m)
// Output:
//   out_dist: array of 32 floats – accumulated approximate distances.
void compute_distances_block(const float* lut, const uint8_t* codes,
                             float* out_dist, int num_subspaces);

} // namespace kscann
