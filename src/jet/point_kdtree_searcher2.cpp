// Copyright (c) 2017 Doyub Kim
//
// I am making my contributions/submissions to this project solely in my
// personal capacity and am not conveying any rights to any intellectual
// property of any third parties.

#include <pch.h>

#include <fbs_helpers.h>

#include <jet/bounding_box2.h>
#include <jet/point_kdtree_searcher2.h>

#include <numeric>

using namespace jet;

PointKdTreeSearcher2::PointKdTreeSearcher2() {}

PointKdTreeSearcher2::PointKdTreeSearcher2(const PointKdTreeSearcher2& other) {
    set(other);
}

void PointKdTreeSearcher2::build(const ConstArrayAccessor1<Vector2D>& points) {
    _tree.build(points);
}

void PointKdTreeSearcher2::forEachNearbyPoint(
    const Vector2D& origin, double radius,
    const ForEachNearbyPointFunc& callback) const {
    _tree.forEachNearbyPoint(origin, radius, callback);
}

bool PointKdTreeSearcher2::hasNearbyPoint(const Vector2D& origin,
                                          double radius) const {
    return _tree.hasNearbyPoint(origin, radius);
}

PointNeighborSearcher2Ptr PointKdTreeSearcher2::clone() const {
    return CLONE_W_CUSTOM_DELETER(PointKdTreeSearcher2);
}

PointKdTreeSearcher2& PointKdTreeSearcher2::operator=(
    const PointKdTreeSearcher2& other) {
    set(other);
    return *this;
}

void PointKdTreeSearcher2::set(const PointKdTreeSearcher2& other) {
    _tree = other._tree;
}

void PointKdTreeSearcher2::serialize(std::vector<uint8_t>* buffer) const {
    // TODO: More
    (void)buffer;
}

void PointKdTreeSearcher2::deserialize(const std::vector<uint8_t>& buffer) {
    // TODO: More
    (void)buffer;
}

PointKdTreeSearcher2::Builder PointKdTreeSearcher2::builder() {
    return Builder{};
}

//

PointKdTreeSearcher2 PointKdTreeSearcher2::Builder::build() const {
    return PointKdTreeSearcher2{};
}

PointKdTreeSearcher2Ptr PointKdTreeSearcher2::Builder::makeShared() const {
    return std::shared_ptr<PointKdTreeSearcher2>(
        new PointKdTreeSearcher2,
        [](PointKdTreeSearcher2* obj) { delete obj; });
}

PointNeighborSearcher2Ptr
PointKdTreeSearcher2::Builder::buildPointNeighborSearcher() const {
    return makeShared();
}
