#pragma once

#include <cstdint>
#include <vector>

namespace kscann {

using Point = std::vector<float>;
using PointId = int64_t;

struct IndexConfig {
    // Base IVF-PQ parameters
    int num_clusters = 100;           // number of IVF centroids
    int num_pq_subspaces = 10;        // m (must divide dimension)
    int num_pq_centroids = 256;       // k* (codebook size per subspace)
    int nprob = 20;                   // number of clusters to probe
    int reorder = 100;                // number of candidates for re‑ranking

    // RAIRS optimisation parameters
    bool use_rairs = false;           // enable redundant assignment + SEIL
    int k_assign = 2;                 // number of clusters each point is assigned to (if use_rairs)
};

} // namespace kscann
