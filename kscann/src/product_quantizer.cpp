#include "kscann/product_quantizer.h"
#include "kscann/kmeans.h"
#include <cassert>
#include <limits>

namespace kscann {

PQCodebook train_pq(const std::vector<Point>& data, int num_subspaces, int num_centroids_per_subspace) {
    int dim = data[0].size();
    assert(dim % num_subspaces == 0);
    int subspace_dim = dim / num_subspaces;

    PQCodebook pq;
    pq.num_subspaces = num_subspaces;
    pq.subspace_dim = subspace_dim;
    pq.centroids.resize(num_subspaces);
    pq.subspace_dims.resize(num_subspaces);

    // Build sub-vectors for each subspace
    std::vector<std::vector<Point>> sub_vectors(num_subspaces);
    for (size_t i = 0; i < data.size(); ++i) {
        for (int s = 0; s < num_subspaces; ++s) {
            Point sub(subspace_dim);
            for (int d = 0; d < subspace_dim; ++d) {
                sub[d] = data[i][s * subspace_dim + d];
            }
            sub_vectors[s].push_back(sub);
        }
    }

    // Train k‑means per subspace
    for (int s = 0; s < num_subspaces; ++s) {
        pq.centroids[s] = kmeans(sub_vectors[s], num_centroids_per_subspace);
        // Record which original dimensions belong to this subspace
        for (int d = 0; d < subspace_dim; ++d) {
            pq.subspace_dims[s].push_back(s * subspace_dim + d);
        }
    }
    return pq;
}

std::vector<uint8_t> encode_pq(const Point& vec, const PQCodebook& pq) {
    std::vector<uint8_t> codes(pq.num_subspaces);
    for (int s = 0; s < pq.num_subspaces; ++s) {
        Point sub(pq.subspace_dim);
        for (int d = 0; d < pq.subspace_dim; ++d) {
            sub[d] = vec[pq.subspace_dims[s][d]];
        }
        // Find nearest centroid
        float best_dist = std::numeric_limits<float>::max();
        int best_idx = 0;
        for (size_t c = 0; c < pq.centroids[s].size(); ++c) {
            float dist = 0.0f;
            for (int d = 0; d < pq.subspace_dim; ++d) {
                float diff = sub[d] - pq.centroids[s][c][d];
                dist += diff * diff;
            }
            if (dist < best_dist) {
                best_dist = dist;
                best_idx = static_cast<int>(c);
            }
        }
        codes[s] = static_cast<uint8_t>(best_idx);
    }
    return codes;
}

} // namespace kscann
