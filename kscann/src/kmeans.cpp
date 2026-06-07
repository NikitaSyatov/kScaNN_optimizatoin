#include "kscann/kmeans.h"
#include <random>
#include <limits>
#include <algorithm>

namespace kscann {

std::vector<Point> kmeans(const std::vector<Point>& data, int num_clusters, int max_iter) {
    int dim = data[0].size();
    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_int_distribution<> dis(0, static_cast<int>(data.size()) - 1);

    // Initialise centroids randomly (choose distinct points)
    std::vector<Point> centroids(num_clusters);
    for (int c = 0; c < num_clusters; ++c) {
        centroids[c] = data[dis(gen)];
    }

    std::vector<int> assignments(data.size());
    for (int iter = 0; iter < max_iter; ++iter) {
        // Assignment
        for (size_t i = 0; i < data.size(); ++i) {
            float best_dist = std::numeric_limits<float>::max();
            int best_c = 0;
            for (int c = 0; c < num_clusters; ++c) {
                float dist = 0.0f;
                for (int d = 0; d < dim; ++d) {
                    float diff = data[i][d] - centroids[c][d];
                    dist += diff * diff;
                }
                if (dist < best_dist) {
                    best_dist = dist;
                    best_c = c;
                }
            }
            assignments[i] = best_c;
        }
        // Update centroids
        std::vector<int> counts(num_clusters, 0);
        std::vector<Point> new_centroids(num_clusters, Point(dim, 0.0f));
        for (size_t i = 0; i < data.size(); ++i) {
            int c = assignments[i];
            counts[c]++;
            for (int d = 0; d < dim; ++d) {
                new_centroids[c][d] += data[i][d];
            }
        }
        for (int c = 0; c < num_clusters; ++c) {
            if (counts[c] > 0) {
                for (int d = 0; d < dim; ++d) {
                    new_centroids[c][d] /= counts[c];
                }
                centroids[c] = new_centroids[c];
            }
        }
    }
    return centroids;
}

} // namespace kscann
