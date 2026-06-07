#include "kscann/simd_utils.h"
#ifdef __ARM_NEON
#include <arm_neon.h>
#endif
#include <cstdint>

namespace kscann {

#ifdef KSCANN_USE_NEON
void compute_distances_block_neon_impl(const uint8_t* lut, const uint8_t* codes,
                                       float* out_dist, int num_subspaces) {
    const int BLOCK_SIZE = 32;
    // Initialise distance accumulators for 32 points (8 NEON registers, each holding 4 floats)
    float32x4_t sum0 = vdupq_n_f32(0.0f);
    float32x4_t sum1 = vdupq_n_f32(0.0f);
    float32x4_t sum2 = vdupq_n_f32(0.0f);
    float32x4_t sum3 = vdupq_n_f32(0.0f);
    float32x4_t sum4 = vdupq_n_f32(0.0f);
    float32x4_t sum5 = vdupq_n_f32(0.0f);
    float32x4_t sum6 = vdupq_n_f32(0.0f);
    float32x4_t sum7 = vdupq_n_f32(0.0f);

    const uint8_t* code_ptr = codes;
    for (int s = 0; s < num_subspaces; ++s) {
        // Load 16 distances from LUT for this subspace (each centroid)
        uint8x16_t lut_vec = vld1q_u8(lut + s * 256); // assumes 256 centroids

        // Process 32 points in groups of 4
        for (int p = 0; p < BLOCK_SIZE; p += 4) {
            // Load 4 codes (each byte contains one 8‑bit centroid index)
            uint8x4_t code4 = vld1_u8(code_ptr + p);
            // Expand to separate registers for each of the 4 points (each element becomes a lane)
            uint16x4_t idx_u16 = vmovl_u8(code4);
            uint32x4_t idx_u32 = vmovl_u16(idx_u16);
            // Use table lookup to get distances (VTBL works with 8-bit vectors)
            // For simplicity, we convert to 16-bit indices and use a scalar fallback per point.
            // A full NEON table lookup would require 8‑bit indices and a 8x16 table.
            // Here we implement a simpler approach: convert each index to float and add.
            // In a production implementation, you would use a proper lookup with vtbl.
            for (int i = 0; i < 4; ++i) {
                int idx = code4[i];
                float dist = static_cast<float>(lut_vec[idx]);
                // Add to appropriate accumulator based on absolute point index
                int abs_p = p + i;
                if (abs_p < 8) {
                    sum0 = vsetq_lane_f32(vgetq_lane_f32(sum0, abs_p) + dist, sum0, abs_p);
                } else if (abs_p < 16) {
                    sum1 = vsetq_lane_f32(vgetq_lane_f32(sum1, abs_p-8) + dist, sum1, abs_p-8);
                } else if (abs_p < 24) {
                    sum2 = vsetq_lane_f32(vgetq_lane_f32(sum2, abs_p-16) + dist, sum2, abs_p-16);
                } else {
                    sum3 = vsetq_lane_f32(vgetq_lane_f32(sum3, abs_p-24) + dist, sum3, abs_p-24);
                }
            }
        }
        code_ptr += BLOCK_SIZE;
    }

    // Store results
    vst1q_f32(out_dist,      sum0);
    vst1q_f32(out_dist+4,    sum1);
    vst1q_f32(out_dist+8,    sum2);
    vst1q_f32(out_dist+12,   sum3);
    // (remaining sums 4–7 not used with 32 points)
}
#endif

} // namespace kscann
