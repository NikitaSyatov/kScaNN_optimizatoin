#include "kscann/ivf_index.h"
#include "kscann/simd_utils.h"
#include "kscann/kmeans.h"
#include "kscann/rair_utils.h"
#include <queue>
#include <algorithm>
#include <limits>
#include <cmath>
#include <unordered_map>

namespace kscann {

// ------------------------------------------------------------------
// Classic build (no redundant assignment, no SEIL)
// ------------------------------------------------------------------
static IVFIndex build_classic(const std::vector<Point>& data,
                              int num_clusters,
                              int num_pq_subspaces,
                              int num_pq_centroids) {
    IVFIndex idx;
    if (data.empty()) return idx;
    idx.dim = static_cast<int>(data[0].size());
    idx.num_clusters = num_clusters;
    
    // Store original data (flat)
    idx.original_data.resize(data.size() * idx.dim);
    for (size_t i = 0; i < data.size(); ++i) {
        std::copy(data[i].begin(), data[i].end(),
                  idx.original_data.begin() + i * idx.dim);
    }

    // 1. Train IVF centroids
    idx.centroids = kmeans(data, num_clusters);

    // 2. Train PQ codebook
    idx.pq = train_pq(data, num_pq_subspaces, num_pq_centroids);

    // 3. Assign points to clusters (single assignment) and encode
    idx.inverted_lists.resize(num_clusters);
    idx.pq_codes.resize(data.size());

    for (size_t i = 0; i < data.size(); ++i) {
        float best_dist = std::numeric_limits<float>::max();
        int best_cluster = 0;
        for (int c = 0; c < num_clusters; ++c) {
            float dist = 0.0f;
            for (int d = 0; d < idx.dim; ++d) {
                float diff = data[i][d] - idx.centroids[c][d];
                dist += diff * diff;
            }
            if (dist < best_dist) {
                best_dist = dist;
                best_cluster = c;
            }
        }
        idx.inverted_lists[best_cluster].push_back(static_cast<PointId>(i));
        idx.pq_codes[i] = encode_pq(data[i], idx.pq);
    }

    // 4. Build SIMD-friendly blocks (32 points per block) for classic layout
    const int BLOCK_SIZE = 32;
    idx.blocks.resize(num_clusters);
    idx.block_sizes.resize(num_clusters, 0);
    for (int c = 0; c < num_clusters; ++c) {
        int num_points = static_cast<int>(idx.inverted_lists[c].size());
        idx.block_sizes[c] = num_points;
        if (num_points == 0) continue;

        std::vector<uint8_t> flat_codes(num_points * idx.pq.num_subspaces);
        for (int i = 0; i < num_points; ++i) {
            PointId pid = idx.inverted_lists[c][i];
            for (int s = 0; s < idx.pq.num_subspaces; ++s) {
                flat_codes[i * idx.pq.num_subspaces + s] = idx.pq_codes[pid][s];
            }
        }

        int num_blocks = (num_points + BLOCK_SIZE - 1) / BLOCK_SIZE;
        idx.blocks[c].resize(num_blocks * BLOCK_SIZE * idx.pq.num_subspaces);
        for (int b = 0; b < num_blocks; ++b) {
            int start = b * BLOCK_SIZE;
            int end = std::min(start + BLOCK_SIZE, num_points);
            for (int i = start; i < end; ++i) {
                for (int s = 0; s < idx.pq.num_subspaces; ++s) {
                    idx.blocks[c][b * BLOCK_SIZE * idx.pq.num_subspaces +
                                    (i - start) * idx.pq.num_subspaces + s] =
                        flat_codes[i * idx.pq.num_subspaces + s];
                }
            }
        }
    }
    return idx;
}

// ------------------------------------------------------------------
// RAIRS build (redundant assignment with AIR + SEIL layout)
// ------------------------------------------------------------------
static IVFIndex build_rairs(const std::vector<Point>& data,
                            int num_clusters,
                            int num_pq_subspaces,
                            int num_pq_centroids,
                            int k_assign) {
    IVFIndex idx;
    idx.use_rairs = true;
    if (data.empty()) return idx;
    idx.dim = data[0].size();
    idx.num_clusters = num_clusters;
    idx.original_data.resize(data.size() * idx.dim);
    for (size_t i = 0; i < data.size(); ++i) {
        std::copy(data[i].begin(), data[i].end(),
                idx.original_data.begin() + i * idx.dim);
    }

    // 1. IVF centroids
    idx.centroids = kmeans(data, num_clusters);
    // 2. PQ codebook
    idx.pq = train_pq(data, num_pq_subspaces, num_pq_centroids);

    // 3. For each point, select top-k_assign clusters using AIR
    std::vector<std::vector<int>> point_to_clusters(data.size());
    for (size_t i = 0; i < data.size(); ++i) {
        point_to_clusters[i] = select_clusters_with_air(data[i], idx.centroids, k_assign);
    }

    // 4. Create cells (one per point) and flat PQ codes
    idx.num_cells = static_cast<int>(data.size());
    idx.cell_pq_codes_flat.resize(idx.num_cells * idx.pq.num_subspaces);
    for (size_t i = 0; i < data.size(); ++i) {
        auto codes = encode_pq(data[i], idx.pq);
        std::copy(codes.begin(), codes.end(),
                  idx.cell_pq_codes_flat.begin() + i * idx.pq.num_subspaces);
    }

    // 5. Build inverted lists (cell IDs) and then block them
    std::vector<std::vector<int>> raw_lists(num_clusters);
    for (size_t i = 0; i < data.size(); ++i) {
        int cell_id = static_cast<int>(i);
        for (int cid : point_to_clusters[i]) {
            raw_lists[cid].push_back(cell_id);
        }
    }

    // 6. Convert raw_lists into blocks of 32 cell IDs
    const int BLOCK_SIZE = 32;
    idx.list_cell_blocks.resize(num_clusters);
    idx.list_num_blocks.resize(num_clusters, 0);
    idx.list_cell_counts.resize(num_clusters, 0);
    for (int c = 0; c < num_clusters; ++c) {
        const auto& lst = raw_lists[c];
        idx.list_cell_counts[c] = static_cast<int>(lst.size());
        int num_blocks = (lst.size() + BLOCK_SIZE - 1) / BLOCK_SIZE;
        idx.list_num_blocks[c] = num_blocks;
        idx.list_cell_blocks[c].resize(num_blocks * BLOCK_SIZE, -1); // pad with -1
        for (size_t i = 0; i < lst.size(); ++i) {
            idx.list_cell_blocks[c][i] = lst[i];
        }
    }

    return idx;
}

// ------------------------------------------------------------------
// Public build_index (dispatches based on config)
// ------------------------------------------------------------------
IVFIndex build_index(const std::vector<Point>& data, const IndexConfig& config) {
    if (config.use_rairs) {
        return build_rairs(data, config.num_clusters,
                           config.num_pq_subspaces,
                           config.num_pq_centroids,
                           config.k_assign);
    } else {
        return build_classic(data, config.num_clusters,
                             config.num_pq_subspaces,
                             config.num_pq_centroids);
    }
}

// ------------------------------------------------------------------
// Search with SEIL support (if cells exist)
// ------------------------------------------------------------------
static std::vector<PointId> search_classic(const IVFIndex& idx, const Point& query,
                                           int k, int nprob, int reorder) {
    int num_subspaces = idx.pq.num_subspaces;
    int num_centroids_per_subspace = static_cast<int>(idx.pq.centroids[0].size());

    // ---- 1. Build float LUT ----
    std::vector<float> lut(num_subspaces * num_centroids_per_subspace);
    for (int s = 0; s < num_subspaces; ++s) {
        Point sub(idx.pq.subspace_dim);
        for (int d = 0; d < idx.pq.subspace_dim; ++d)
            sub[d] = query[idx.pq.subspace_dims[s][d]];
        for (int c = 0; c < num_centroids_per_subspace; ++c) {
            float dist = 0.0f;
            for (int d = 0; d < idx.pq.subspace_dim; ++d) {
                float diff = sub[d] - idx.pq.centroids[s][c][d];
                dist += diff * diff;
            }
            lut[s * num_centroids_per_subspace + c] = dist;
        }
    }

    // ---- 2. Select clusters to probe ----
    struct DistToCluster { float dist; int cluster_id; };
    std::vector<DistToCluster> cluster_dists(idx.num_clusters);
    for (int c = 0; c < idx.num_clusters; ++c) {
        float dist = 0.0f;
        for (int d = 0; d < idx.dim; ++d) {
            float diff = query[d] - idx.centroids[c][d];
            dist += diff * diff;
        }
        cluster_dists[c] = {dist, c};
    }
    nprob = std::min(nprob, idx.num_clusters);
    std::partial_sort(cluster_dists.begin(), cluster_dists.begin() + nprob,
                      cluster_dists.end(),
                      [](const auto& a, const auto& b) { return a.dist < b.dist; });

    // ---- 3. Collect top `reorder` approximate candidates ----
    using Candidate = std::pair<float, PointId>;
    auto cmp = [](const Candidate& a, const Candidate& b) { return a.first < b.first; };
    std::priority_queue<Candidate, std::vector<Candidate>, decltype(cmp)> heap(cmp);

    for (int i = 0; i < nprob; ++i) {
        int cid = cluster_dists[i].cluster_id;
        const auto& point_ids = idx.inverted_lists[cid];
        for (PointId pid : point_ids) {
            const auto& codes = idx.pq_codes[pid];
            float dist = 0.0f;
            for (int s = 0; s < num_subspaces; ++s) {
                uint8_t code = codes[s];
                dist += lut[s * num_centroids_per_subspace + code];
            }
            if (static_cast<int>(heap.size()) < reorder) {
                heap.emplace(dist, pid);
            } else if (dist < heap.top().first) {
                heap.pop();
                heap.emplace(dist, pid);
            }
        }
    }

    // ---- 4. Exact re‑ranking for the candidates ----
    std::vector<Candidate> candidates;
    candidates.reserve(heap.size());
    while (!heap.empty()) {
        candidates.push_back(heap.top());
        heap.pop();
    }
    // Compute exact L2 distances
    for (auto& cand : candidates) {
        PointId pid = cand.second;
        const float* vec = idx.original_data.data() + pid * idx.dim;
        float exact_dist = 0.0f;
        for (int d = 0; d < idx.dim; ++d) {
            float diff = query[d] - vec[d];
            exact_dist += diff * diff;
        }
        cand.first = exact_dist;
    }
    // Sort by exact distance
    std::sort(candidates.begin(), candidates.end(),
              [](const Candidate& a, const Candidate& b) { return a.first < b.first; });

    // ---- 5. Return top‑k ----
    std::vector<PointId> result;
    result.reserve(k);
    for (int i = 0; i < std::min(k, static_cast<int>(candidates.size())); ++i) {
        result.push_back(candidates[i].second);
    }
    return result;
}

static std::vector<PointId> search_seil(const IVFIndex& idx, const Point& query,
                                        int k, int nprob, int reorder) {
    int num_subspaces = idx.pq.num_subspaces;
    int num_centroids_per_subspace = static_cast<int>(idx.pq.centroids[0].size());

    // ---------- 1. Build float LUT ----------
    std::vector<float> lut(num_subspaces * num_centroids_per_subspace);
    for (int s = 0; s < num_subspaces; ++s) {
        Point sub(idx.pq.subspace_dim);
        for (int d = 0; d < idx.pq.subspace_dim; ++d)
            sub[d] = query[idx.pq.subspace_dims[s][d]];
        for (int c = 0; c < num_centroids_per_subspace; ++c) {
            float dist = 0.0f;
            for (int d = 0; d < idx.pq.subspace_dim; ++d) {
                float diff = sub[d] - idx.pq.centroids[s][c][d];
                dist += diff * diff;
            }
            lut[s * num_centroids_per_subspace + c] = dist;
        }
    }

    // ---------- 2. Select clusters to probe ----------
    struct DistToCluster { float dist; int cluster_id; };
    std::vector<DistToCluster> cluster_dists(idx.num_clusters);
    for (int c = 0; c < idx.num_clusters; ++c) {
        float dist = 0.0f;
        for (int d = 0; d < idx.dim; ++d) {
            float diff = query[d] - idx.centroids[c][d];
            dist += diff * diff;
        }
        cluster_dists[c] = {dist, c};
    }
    nprob = std::min(nprob, idx.num_clusters);
    std::partial_sort(cluster_dists.begin(), cluster_dists.begin() + nprob,
                      cluster_dists.end(),
                      [](const auto& a, const auto& b) { return a.dist < b.dist; });

    // ---------- 3. Collect top `reorder` approximate candidates (SEIL layout) ----------
    using Candidate = std::pair<float, PointId>;
    auto cmp = [](const Candidate& a, const Candidate& b) { return a.first < b.first; };
    std::priority_queue<Candidate, std::vector<Candidate>, decltype(cmp)> heap(cmp);

    // Flat cache to avoid recomputing same cell (though we will recompute for simplicity)
    // For clarity, we skip the cache and recompute each cell (cells are visited multiple times but we keep the block gathering)
    const int BLOCK_SIZE = 32;
    std::vector<uint8_t> block_codes(BLOCK_SIZE * num_subspaces);

    for (int i = 0; i < nprob; ++i) {
        int cid = cluster_dists[i].cluster_id;
        const auto& blocks = idx.list_cell_blocks[cid];
        int num_blocks = idx.list_num_blocks[cid];
        int num_cells_in_cluster = idx.list_cell_counts[cid];

        for (int b = 0; b < num_blocks; ++b) {
            int cells_in_block = std::min(BLOCK_SIZE, num_cells_in_cluster - b * BLOCK_SIZE);
            // Gather PQ codes for this block
            for (int j = 0; j < cells_in_block; ++j) {
                int cell_id = blocks[b * BLOCK_SIZE + j];
                if (cell_id == -1) continue;
                const uint8_t* src = idx.cell_pq_codes_flat.data() + cell_id * num_subspaces;
                std::copy(src, src + num_subspaces, block_codes.begin() + j * num_subspaces);
            }
            // Compute approximate distances for all cells in the block
            for (int j = 0; j < cells_in_block; ++j) {
                int cell_id = blocks[b * BLOCK_SIZE + j];
                if (cell_id == -1) continue;
                float dist = 0.0f;
                const uint8_t* codes = block_codes.data() + j * num_subspaces;
                for (int s = 0; s < num_subspaces; ++s) {
                    dist += lut[s * num_centroids_per_subspace + codes[s]];
                }
                if (static_cast<int>(heap.size()) < reorder) {
                    heap.emplace(dist, static_cast<PointId>(cell_id));
                } else if (dist < heap.top().first) {
                    heap.pop();
                    heap.emplace(dist, static_cast<PointId>(cell_id));
                }
            }
        }
    }

    // ---------- 4. Exact re‑ranking ----------
    std::vector<Candidate> candidates;
    candidates.reserve(heap.size());
    while (!heap.empty()) {
        candidates.push_back(heap.top());
        heap.pop();
    }
    // Compute exact L2 distances
    for (auto& cand : candidates) {
        PointId pid = cand.second;
        const float* vec = idx.original_data.data() + pid * idx.dim;
        float exact_dist = 0.0f;
        for (int d = 0; d < idx.dim; ++d) {
            float diff = query[d] - vec[d];
            exact_dist += diff * diff;
        }
        cand.first = exact_dist;
    }
    std::sort(candidates.begin(), candidates.end(),
              [](const Candidate& a, const Candidate& b) { return a.first < b.first; });

    // ---------- 5. Return top‑k ----------
    std::vector<PointId> result;
    result.reserve(k);
    for (int i = 0; i < std::min(k, static_cast<int>(candidates.size())); ++i) {
        result.push_back(candidates[i].second);
    }
    return result;
}

// Public search dispatcher
std::vector<PointId> search(const IVFIndex& idx, const Point& query,
                            int k, int nprob, int reorder) {
    if (!idx.use_rairs) {
        return search_classic(idx, query, k, nprob, reorder);
    } else {
        return search_seil(idx, query, k, nprob, reorder);
    }
}

} // namespace kscann