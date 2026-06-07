#include "kscann/rair_utils.h"
#include <cmath>
#include <queue>
#include <algorithm>

namespace kscann {

// Euclidean distance squared (used internally)
static float l2_squared(const Point& a, const Point& b) {
    float sum = 0.0f;
    for (size_t i = 0; i < a.size(); ++i) {
        float diff = a[i] - b[i];
        sum += diff * diff;
    }
    return sum;
}

// Dot product
static float dot(const Point& a, const Point& b) {
    float sum = 0.0f;
    for (size_t i = 0; i < a.size(); ++i) sum += a[i] * b[i];
    return sum;
}

float compute_air(const Point& point,
                  const Point& primary_centroid,
                  const Point& secondary_centroid) {
    // Distances (squared Euclidean)
    float d1 = l2_squared(point, primary_centroid);
    float d2 = l2_squared(point, secondary_centroid);
    // Directional benefit: higher dot product means point is oriented similarly to secondary centroid
    float dir = dot(point, secondary_centroid);
    // Residual: absolute difference in distances
    float residual = std::abs(std::sqrt(d1) - std::sqrt(d2));
    // AIR formula from paper (simplified – adjust if needed)
    // Larger AIR means better candidate for redundant assignment.
    return (d2 - dir) * residual;
}

std::vector<int> select_clusters_with_air(const Point& point,
                                          const std::vector<Point>& centroids,
                                          int k_assign) {
    int num_centroids = static_cast<int>(centroids.size());
    if (k_assign <= 0 || k_assign > num_centroids) k_assign = num_centroids;

    // Compute distances to all centroids
    struct DistIdx { float dist; int idx; };
    std::vector<DistIdx> dists(num_centroids);
    for (int i = 0; i < num_centroids; ++i) {
        dists[i].dist = l2_squared(point, centroids[i]);
        dists[i].idx = i;
    }
    // Sort by distance to get primary (closest)
    std::sort(dists.begin(), dists.end(),
              [](const DistIdx& a, const DistIdx& b) { return a.dist < b.dist; });

    std::vector<int> selected;
    selected.push_back(dists[0].idx); // primary centroid

    if (k_assign == 1) return selected;

    // For the remaining k_assign-1, score using AIR with primary as reference
    const Point& primary_centroid = centroids[dists[0].idx];
    std::vector<std::pair<float, int>> air_scores;
    for (int i = 1; i < num_centroids; ++i) {
        float air = compute_air(point, primary_centroid, centroids[dists[i].idx]);
        air_scores.emplace_back(air, dists[i].idx);
    }
    // Sort by AIR descending (higher is better)
    std::sort(air_scores.begin(), air_scores.end(),
              [](const auto& a, const auto& b) { return a.first > b.first; });

    for (int i = 0; i < k_assign - 1 && i < static_cast<int>(air_scores.size()); ++i) {
        selected.push_back(air_scores[i].second);
    }
    return selected;
}

} // namespace kscann