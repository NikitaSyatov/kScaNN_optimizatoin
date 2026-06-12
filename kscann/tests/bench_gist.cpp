#include "kscann/kscann.h"
#include <iostream>
#include <fstream>
#include <chrono>
#include <vector>
#include <string>
#include <algorithm>
#include <set>
#include <numeric>
#include <cmath>

// .fvecs loader
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

// Load ground truth from .ivecs file (e.g., gist_groundtruth.ivecs)
// Format: for each query: 4-byte integer K (e.g., 100), then K integers (neighbor IDs)
std::vector<std::vector<int64_t>> load_ivecs_groundtruth(const std::string& filename, int num_queries, int& k_gt) {
    std::ifstream in(filename, std::ios::binary);
    if (!in) return {};
    std::vector<std::vector<int64_t>> gt;
    int queries_read = 0;
    while (in.peek() != EOF && queries_read < num_queries) {
        int k;
        in.read(reinterpret_cast<char*>(&k), sizeof(int));
        if (queries_read == 0) k_gt = k;
        std::vector<int64_t> ids(k);
        std::vector<int32_t> ids32(k);
        in.read(reinterpret_cast<char*>(ids32.data()), k * sizeof(int32_t));
        for (int i = 0; i < k; ++i) ids[i] = static_cast<int64_t>(ids32[i]);
        gt.push_back(ids);
        ++queries_read;
    }
    return gt;
}

float compute_recall(const std::vector<int64_t>& approx, const std::vector<int64_t>& exact, int k) {
    std::set<int64_t> exact_set(exact.begin(), exact.begin() + k);
    int intersection = 0;
    for (int i = 0; i < std::min(k, (int)approx.size()); ++i) {
        if (exact_set.count(approx[i])) ++intersection;
    }
    return static_cast<float>(intersection) / static_cast<float>(k);
}

int main(int argc, char** argv) {
    if (argc < 4) {
        std::cerr << "Usage: " << argv[0] << " <base.fvecs> <query.fvecs> <groundtruth.ivecs> [rairs] [num_queries]\n";
        return 1;
    }
    std::string base_file = argv[1];
    std::string query_file = argv[2];
    std::string gt_file = argv[3];
    bool use_rairs = (argc >= 5 && std::string(argv[4]) == "rairs");
    int num_queries = (argc >= 6) ? std::stoi(argv[5]) : 100;

    size_t num_base = 0;
    int dim = 0;
    auto base = load_fvecs(base_file, num_base, dim);
    if (num_base == 0) {
        std::cerr << "Failed to load base vectors.\n";
        return 1;
    }
    std::cout << "Loaded " << num_base << " base vectors, dim=" << dim << "\n";

    size_t num_query_available = 0;
    int query_dim = 0;
    auto query_data = load_fvecs(query_file, num_query_available, query_dim);
    if (num_query_available == 0) {
        std::cerr << "Failed to load query vectors.\n";
        return 1;
    }
    num_queries = std::min(num_queries, (int)num_query_available);
    std::cout << "Using " << num_queries << " queries\n";

    // Prepare query vectors
    std::vector<std::vector<float>> queries(num_queries);
    for (int i = 0; i < num_queries; ++i) {
        queries[i].resize(dim);
        std::copy(query_data.begin() + i * dim, query_data.begin() + (i+1) * dim, queries[i].begin());
    }

    // Load ground truth from .ivecs
    int k_gt = 0;
    auto exact_results = load_ivecs_groundtruth(gt_file, num_queries, k_gt);
    if (exact_results.empty()) {
        std::cerr << "Failed to load ground truth from " << gt_file << "\n";
        return 1;
    }
    std::cout << "Loaded ground truth with k=" << k_gt << " neighbours per query\n";

    // Index configuration (adjust for GIST1M)
    kscann::IndexConfig cfg;
    cfg.num_clusters = 1000;           // 1M points → 1000 clusters
    cfg.num_pq_subspaces = 16;         // 960 / 16 = 60 dims per subspace
    cfg.num_pq_centroids = 256;
    cfg.nprob = 50;                    // probe 5% of clusters
    cfg.reorder = 500;                 // re-rank 500 candidates
    cfg.use_rairs = use_rairs;
    if (use_rairs) {
        cfg.k_assign = 2;
        std::cout << "RAIRS enabled, k_assign=" << cfg.k_assign << "\n";
    } else {
        std::cout << "Baseline IVF-PQ\n";
    }

    kscann::Index index(cfg);
    auto start_build = std::chrono::high_resolution_clock::now();
    index.build(base, num_base, dim);
    auto end_build = std::chrono::high_resolution_clock::now();
    double build_sec = std::chrono::duration<double>(end_build - start_build).count();
    std::cout << "Index built in " << build_sec << " s\n";

    // Search
    int k = 10;  // recall@10
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