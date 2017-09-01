// Copyright (c) 2017 Doyub Kim
//
// I am making my contributions/submissions to this project solely in my
// personal capacity and am not conveying any rights to any intellectual
// property of any third parties.

#include <pch.h>

#include <fbs_helpers.h>

#include <jet/bounding_box3.h>
#include <jet/point_kdtree_searcher3.h>

#include <numeric>

using namespace jet;

PointKdTreeSearcher3::PointKdTreeSearcher3() {}

PointKdTreeSearcher3::PointKdTreeSearcher3(const PointKdTreeSearcher3& other) {
    set(other);
}

void PointKdTreeSearcher3::build(const ConstArrayAccessor1<Vector3D>& points) {
    _tree.build(points);
}

void PointKdTreeSearcher3::forEachNearbyPoint(
    const Vector3D& origin, double radius,
    const ForEachNearbyPointFunc& callback) const {
    _tree.forEachNearbyPoint(origin, radius, callback);
}

bool PointKdTreeSearcher3::hasNearbyPoint(const Vector3D& origin,
                                          double radius) const {
    return _tree.hasNearbyPoint(origin, radius);
}

PointNeighborSearcher3Ptr PointKdTreeSearcher3::clone() const {
    return CLONE_W_CUSTOM_DELETER(PointKdTreeSearcher3);
}

PointKdTreeSearcher3& PointKdTreeSearcher3::operator=(
    const PointKdTreeSearcher3& other) {
    set(other);
    return *this;
}

void PointKdTreeSearcher3::set(const PointKdTreeSearcher3& other) {
    _tree = other._tree;
}

void PointKdTreeSearcher3::serialize(std::vector<uint8_t>* buffer) const {
    // TODO: More
    (void)buffer;
}

void PointKdTreeSearcher3::deserialize(const std::vector<uint8_t>& buffer) {
    // TODO: More
    (void)buffer;
}

PointKdTreeSearcher3::Builder PointKdTreeSearcher3::builder() {
    return Builder{};
}

//

PointKdTreeSearcher3 PointKdTreeSearcher3::Builder::build() const {
    return PointKdTreeSearcher3{};
}

PointKdTreeSearcher3Ptr PointKdTreeSearcher3::Builder::makeShared() const {
    return std::shared_ptr<PointKdTreeSearcher3>(
        new PointKdTreeSearcher3,
        [](PointKdTreeSearcher3* obj) { delete obj; });
}

PointNeighborSearcher3Ptr
PointKdTreeSearcher3::Builder::buildPointNeighborSearcher() const {
    return makeShared();
}
