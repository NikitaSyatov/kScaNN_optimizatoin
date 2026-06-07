#pragma once

#include "types.h"
#include <vector>
#include <memory>

namespace kscann {

class Index {
public:
    explicit Index(const IndexConfig& config = IndexConfig());
    ~Index();

    // Build index from a flat array of floats (row-major, num_points * dim)
    void build(const std::vector<float>& data, size_t num_points, int dim);

    // Search for k nearest neighbours, returns point IDs (indices in the build data)
    std::vector<int64_t> search(const std::vector<float>& query, int k) const;

private:
    struct Impl;
    std::unique_ptr<Impl> impl_;
};

} // namespace kscann
