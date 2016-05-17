// Copyright (c) 2016 Doyub Kim

#include <pch.h>
#include <jet/point_simple_list_searcher2.h>
#include <algorithm>

using namespace jet;

PointSimpleListSearcher2::PointSimpleListSearcher2() {
}

void PointSimpleListSearcher2::build(
    const ConstArrayAccessor1<Vector2D>& points) {
    _points.resize(points.size());
    std::copy(points.data(), points.data() + points.size(), _points.begin());
}

void PointSimpleListSearcher2::forEachNearbyPoint(
    const Vector2D& origin,
    double radius,
    const ForEachNearbyPointFunc& callback) const {
    double radiusSquared = radius * radius;
    for (size_t i = 0; i < _points.size(); ++i) {
        Vector2D r = _points[i] - origin;
        double distanceSquared = r.dot(r);
        if (distanceSquared <= radiusSquared) {
            callback(i, _points[i]);
        }
    }
}

bool PointSimpleListSearcher2::hasNearbyPoint(
    const Vector2D& origin,
    double radius) const {
    double radiusSquared = radius * radius;
    for (size_t i = 0; i < _points.size(); ++i) {
        Vector2D r = _points[i] - origin;
        double distanceSquared = r.dot(r);
        if (distanceSquared <= radiusSquared) {
            return true;
        }
    }

    return false;
}
