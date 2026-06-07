#include "kscann/simd_utils.h"
#include <cstdint>

namespace kscann {

// Fallback scalar implementation for distance computation.
// It processes points one by one (without SIMD).
void compute_distances_block_scalar(const float* lut, const uint8_t* codes,
                                    float* out_dist, int num_subspaces) {
    const int BLOCK_SIZE = 32;
    for (int p = 0; p < BLOCK_SIZE; ++p) {
        float dist = 0.0f;
        const uint8_t* point_codes = codes + p * num_subspaces;
        for (int s = 0; s < num_subspaces; ++s) {
            int centroid = point_codes[s];
            dist += static_cast<float>(lut[s * 256 + centroid]); // assumes 256 centroids
        }
        out_dist[p] = dist;
    }
}

// Dispatch function – if NEON is available at compile time, we call the NEON version.
// Otherwise we fall back to scalar.
void compute_distances_block(const float* lut, const uint8_t* codes,
                             float* out_dist, int num_subspaces) {
#ifdef KSCANN_USE_NEON
    // The NEON implementation is provided in simd_neon.cpp.
    // We forward declare it here (the actual function is extern "C" or in a namespace).
    void compute_distances_block_neon_impl(const uint8_t* lut, const uint8_t* codes,
                                           float* out_dist, int num_subspaces);
    compute_distances_block_neon_impl(lut, codes, out_dist, num_subspaces);
#else
    compute_distances_block_scalar(lut, codes, out_dist, num_subspaces);
#endif
}

} // namespace kscann
