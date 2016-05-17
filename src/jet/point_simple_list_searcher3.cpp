// Copyright (c) 2016 Doyub Kim

#include <pch.h>
#include <jet/point_simple_list_searcher3.h>
#include <algorithm>

using namespace jet;

PointSimpleListSearcher3::PointSimpleListSearcher3() {
}

void PointSimpleListSearcher3::build(
    const ConstArrayAccessor1<Vector3D>& points) {
    _points.resize(points.size());
    std::copy(points.data(), points.data() + points.size(), _points.begin());
}

void PointSimpleListSearcher3::forEachNearbyPoint(
    const Vector3D& origin,
    double radius,
    const std::function<void(size_t, const Vector3D&)>& callback) const {
    double radiusSquared = radius * radius;
    for (size_t i = 0; i < _points.size(); ++i) {
        Vector3D r = _points[i] - origin;
        double distanceSquared = r.dot(r);
        if (distanceSquared <= radiusSquared) {
            callback(i, _points[i]);
        }
    }
}

bool PointSimpleListSearcher3::hasNearbyPoint(
    const Vector3D& origin,
    double radius) const {
    double radiusSquared = radius * radius;
    for (size_t i = 0; i < _points.size(); ++i) {
        Vector3D r = _points[i] - origin;
        double distanceSquared = r.dot(r);
        if (distanceSquared <= radiusSquared) {
            return true;
        }
    }

    return false;
}
