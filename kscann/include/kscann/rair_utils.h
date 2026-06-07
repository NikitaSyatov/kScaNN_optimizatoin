#pragma once

#include "types.h"
#include <vector>
#include <unordered_map>

namespace kscann {

// Compute AIR score for assigning a point to a secondary centroid
// given its primary centroid and the secondary candidate.
float compute_air(const Point& point,
                  const Point& primary_centroid,
                  const Point& secondary_centroid);

// Given a point and all centroids, return the indices of the top k_assign centroids
// using AIR for the secondary ones (first is always the closest by L2 distance).
std::vector<int> select_clusters_with_air(const Point& point,
                                          const std::vector<Point>& centroids,
                                          int k_assign);

// SEIL cache: per-query, stores computed distances for cells
class SeilQueryCache {
public:
    void clear() { cache_.clear(); }
    bool get(int cell_id, float& dist) const {
        auto it = cache_.find(cell_id);
        if (it != cache_.end()) {
            dist = it->second;
            return true;
        }
        return false;
    }
    void set(int cell_id, float dist) { cache_[cell_id] = dist; }
private:
    mutable std::unordered_map<int, float> cache_;
};

} // namespace kscann