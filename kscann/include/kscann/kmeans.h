#pragma once

#include "types.h"
#include <vector>

namespace kscann {

// Run k‑means on data (list of points) and return centroids.
std::vector<Point> kmeans(const std::vector<Point>& data, int num_clusters, int max_iter = 20);

} // namespace kscann