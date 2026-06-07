#pragma once

#include "types.h"
#include <cstdint>

namespace kscann {

struct PQCodebook {
    int num_subspaces;                // m
    int subspace_dim;                 // d' = d / m
    std::vector<std::vector<Point>> centroids; // [subspace][centroid][subspace_dim]
    std::vector<std::vector<int>> subspace_dims; // which original dimensions belong to each subspace
};

// Train a PQ codebook on the given dataset.
PQCodebook train_pq(const std::vector<Point>& data, int num_subspaces, int num_centroids_per_subspace);

// Encode a single vector using the trained codebook.
std::vector<uint8_t> encode_pq(const Point& vec, const PQCodebook& pq);

} // namespace kscann