#include "kscann/kscann.h"
#include <iostream>
#include <fstream>
#include <chrono>
#include <vector>
#include <string>
#include <algorithm>
#include <set>
#include <numeric>

// .fvecs loader (same as before)
std::vector<float> load_fvecs(const std::string& filename, size_t& num_points, int& dim) {
    std::ifstream in(filename, std::ios::binary);
    if (!in) return {};
    num_points = 0;
    std::vector<float> data;
    while (in.peek() != EOF) {
        int d;
        in.read(reinterpret_cast<char*>(&d), sizeof(int));
        if (num_points == 0) dim = d;
        else if (d != dim) break;
        std::vector<float> vec(d);
        in.read(reinterpret_cast<char*>(vec.data()), d * sizeof(float));
        data.insert(data.end(), vec.begin(), vec.end());
        ++num_points;
    }
    return data;
}

// Exact top‑k (brute‑force L2)
std::vector<int64_t> exact_topk(const std::vector<float>& data, size_t num_points, int dim,
                                const std::vector<float>& query, int k) {
    using DistIdx = std::pair<float, int64_t>;
    std::vector<DistIdx> dists(num_points);
    for (size_t i = 0; i < num_points; ++i) {
        float dist = 0.0f;
        const float* vec = data.data() + i * dim;
        for (int d = 0; d < dim; ++d) {
            float diff = query[d] - vec[d];
            dist += diff * diff;
        }
        dists[i] = {dist, static_cast<int64_t>(i)};
    }
    std::partial_sort(dists.begin(), dists.begin() + k, dists.end());
    std::vector<int64_t> result(k);
    for (int i = 0; i < k; ++i) result[i] = dists[i].second;
    return result;
}

float compute_recall(const std::vector<int64_t>& approx, const std::vector<int64_t>& exact, int k) {
    std::set<int64_t> exact_set(exact.begin(), exact.end());
    int intersection = 0;
    for (int i = 0; i < std::min(k, (int)approx.size()); ++i) {
        if (exact_set.count(approx[i])) ++intersection;
    }
    return static_cast<float>(intersection) / static_cast<float>(k);
}

int main(int argc, char** argv) {
    if (argc < 2) {
        std::cerr << "Usage: " << argv[0] << " <path_to_mnist_base.fvecs> [rairs] [num_queries]\n";
        return 1;
    }
    std::string base_file = argv[1];
    bool use_rairs = (argc >= 3 && std::string(argv[2]) == "rairs");
    int num_queries = (argc >= 4) ? std::stoi(argv[3]) : 100;  // limit for speed

    size_t num_base = 0;
    int dim = 0;
    auto base = load_fvecs(base_file, num_base, dim);
    if (num_base == 0) {
        std::cerr << "Failed to load base vectors.\n";
        return 1;
    }
    std::cout << "Loaded " << num_base << " training vectors of dimension " << dim << "\n";
    num_queries = std::min(num_queries, (int)num_base);

    // Use first `num_queries` points as queries (you can also load a separate query file)
    std::vector<std::vector<float>> queries(num_queries);
    for (int i = 0; i < num_queries; ++i) {
        queries[i].resize(dim);
        std::copy(base.begin() + i * dim, base.begin() + (i+1) * dim, queries[i].begin());
    }

    // Configure index
    kscann::IndexConfig cfg;
    cfg.num_clusters = 100;           // MNIST 60k points → 100 clusters
    cfg.num_pq_subspaces = 16;        // 784 / 16 = 49 dims per subspace (must divide)
    cfg.num_pq_centroids = 256;
    cfg.nprob = 20;
    cfg.reorder = 200;
    cfg.use_rairs = use_rairs;
    if (use_rairs) {
        cfg.k_assign = 2;
        std::cout << "RAIRS enabled, k_assign = " << cfg.k_assign << "\n";
    } else {
        std::cout << "Baseline IVF-PQ (no RAIRS)\n";
    }

    kscann::Index index(cfg);
    auto start_build = std::chrono::high_resolution_clock::now();
    index.build(base, num_base, dim);
    auto end_build = std::chrono::high_resolution_clock::now();
    double build_sec = std::chrono::duration<double>(end_build - start_build).count();
    std::cout << "Index built in " << build_sec << " s\n";

    // Pre‑compute exact ground truth (only once)
    std::cout << "Computing exact ground truth (brute‑force) for " << num_queries << " queries...\n";
    std::vector<std::vector<int64_t>> exact_results(num_queries);
    int k = 10;
    for (int i = 0; i < num_queries; ++i) {
        exact_results[i] = exact_topk(base, num_base, dim, queries[i], k);
    }

    // Search and measure
    double total_time_sec = 0.0;
    std::vector<float> recalls(num_queries);
    for (int i = 0; i < num_queries; ++i) {
        auto qstart = std::chrono::high_resolution_clock::now();
        auto approx = index.search(queries[i], k);
        auto qend = std::chrono::high_resolution_clock::now();
        total_time_sec += std::chrono::duration<double>(qend - qstart).count();
        recalls[i] = compute_recall(approx, exact_results[i], k);
    }
    double avg_recall = std::accumulate(recalls.begin(), recalls.end(), 0.0) / num_queries;
    double rps = static_cast<double>(num_queries) / total_time_sec;

    std::cout << "Total queries: " << num_queries << "\n";
    std::cout << "Total search time: " << total_time_sec << " s\n";
    std::cout << "RPS (queries per second): " << rps << "\n";
    std::cout << "Recall@" << k << ": " << avg_recall * 100.0 << "%\n";

    return 0;
}