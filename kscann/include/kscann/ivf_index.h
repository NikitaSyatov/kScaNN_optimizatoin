#pragma once

#include "types.h"
#include "product_quantizer.h"
#include <vector>
#include <cstdint>

namespace kscann {

struct Cell {
    PointId point_id;
    std::vector<uint8_t> pq_codes;
};

struct IVFIndex {
    int dim = 0;
    int num_clusters = 0;
    std::vector<float> original_data;
    std::vector<Point> centroids;

    // Classic (no RAIRS)
    std::vector<std::vector<PointId>> inverted_lists;
    std::vector<std::vector<uint8_t>> pq_codes;
    std::vector<std::vector<uint8_t>> blocks;   // SIMD blocks for classic
    std::vector<int> block_sizes;

    // RAIRS + SEIL (optimised for speed)
    bool use_rairs = false;
    int num_cells = 0;
    // Flat PQ codes for all cells: shape [num_cells * num_subspaces]
    std::vector<uint8_t> cell_pq_codes_flat;
    // For each cluster, a list of cell indices (blocked into 32)
    std::vector<std::vector<int>> list_cell_blocks;   // each block: up to 32 cell IDs
    std::vector<int> list_num_blocks;                 // number of blocks per cluster
    std::vector<int> list_cell_counts;                // total cells per cluster

    // Shared
    PQCodebook pq;
};

IVFIndex build_index(const std::vector<Point>& data, const IndexConfig& config);
std::vector<PointId> search(const IVFIndex& idx, const Point& query,
                            int k, int nprob, int reorder);

} // namespace kscann