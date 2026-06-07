#include "kscann/ivf_index.h"
#include "kscann/types.h"
#include <iostream>
#include <cassert>
#include <algorithm>
#include <cstdlib>   // for std::srand

using namespace kscann;

int main() {
    // Fixed seed for reproducible k‑means
    std::srand(42);

    // Create synthetic dataset: 200 points, 20 dimensions
    const int N = 200;
    const int D = 20;
    std::vector<Point> data(N, Point(D, 0.0f));
    for (int i = 0; i < N; ++i) {
        for (int d = 0; d < D; ++d) {
            data[i][d] = static_cast<float>(i + d);
        }
    }

    // Use more generous parameters to ensure the query point is retrieved
    IndexConfig cfg;
    cfg.num_clusters = 10;
    cfg.num_pq_subspaces = 5;      // D=20 → 4 dimensions per subspace
    cfg.num_pq_centroids = 256;    // maximum precision
    cfg.nprob = 10;                // probe enough clusters
    cfg.reorder = 200;             // re‑rank many candidates
    cfg.use_rairs = false;         // classic IVF‑PQ

    IVFIndex idx = build_index(data, cfg);

    // Query: the first point
    Point query = data[0];
    int k = 5;
    auto result = search(idx, query, k, cfg.nprob, cfg.reorder);

    // Verify that point 0 (the query itself) is among the results
    bool found_self = std::find(result.begin(), result.end(), 0) != result.end();
    assert(found_self);

    std::cout << "test_ivf_pq passed.\n";
    return 0;
}