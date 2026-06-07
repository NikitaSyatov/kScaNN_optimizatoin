#include "kscann/kscann.h"
#include "kscann/ivf_index.h"
#include <memory>
#include <vector>
#include <cstring>

namespace kscann {

struct Index::Impl {
    IVFIndex idx;
    IndexConfig cfg;
};

Index::Index(const IndexConfig& config) : impl_(std::make_unique<Impl>()) {
    impl_->cfg = config;
}

Index::~Index() = default;

void Index::build(const std::vector<float>& data, size_t num_points, int dim) {
    std::vector<Point> points(num_points);
    for (size_t i = 0; i < num_points; ++i) {
        points[i].resize(dim);
        std::memcpy(points[i].data(), data.data() + i * dim, dim * sizeof(float));
    }
    impl_->idx = build_index(points, impl_->cfg);
}

std::vector<int64_t> Index::search(const std::vector<float>& query, int k) const {
    Point q(query.begin(), query.end());
    return kscann::search(impl_->idx, q, k, impl_->cfg.nprob, impl_->cfg.reorder);
}

} // namespace kscann