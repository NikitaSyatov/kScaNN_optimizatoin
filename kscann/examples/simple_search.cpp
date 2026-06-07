#include "kscann/kscann.h"
#include <iostream>
#include <random>

int main() {
    // Synthetic dataset
    const int num_points = 1000;
    const int dim = 100;
    std::random_device rd;
    std::mt19937 gen(rd());
    std::normal_distribution<float> dist(0.0f, 1.0f);

    std::vector<float> data(num_points * dim);
    for (int i = 0; i < num_points; ++i) {
        for (int d = 0; d < dim; ++d) {
            data[i * dim + d] = dist(gen);
        }
    }

    // Build index
    kscann::IndexConfig cfg;
    cfg.num_clusters = 50;
    cfg.num_pq_subspaces = 10;
    cfg.num_pq_centroids = 256;
    cfg.nprob = 10;
    cfg.reorder = 50;
    kscann::Index index(cfg);
    index.build(data, num_points, dim);

    // Query: first point
    std::vector<float> query(data.begin(), data.begin() + dim);
    auto neighbors = index.search(query, 5);
    std::cout << "Nearest neighbour IDs: ";
    for (auto id : neighbors) std::cout << id << " ";
    std::cout << std::endl;

    return 0;
}
