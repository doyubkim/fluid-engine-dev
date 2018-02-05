// Copyright (c) 2018 Doyub Kim
//
// I am making my contributions/submissions to this project solely in my
// personal capacity and am not conveying any rights to any intellectual
// property of any third parties.

#include <pch.h>
#include <fbs_helpers.h>
#include <generated/point_simple_list_searcher2_generated.h>

#include <jet/point_simple_list_searcher2.h>

#include <algorithm>
#include <vector>

using namespace jet;

PointSimpleListSearcher2::PointSimpleListSearcher2() {
}

PointSimpleListSearcher2::PointSimpleListSearcher2(
    const PointSimpleListSearcher2& other) {
    set(other);
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

PointNeighborSearcher2Ptr PointSimpleListSearcher2::clone() const {
    return CLONE_W_CUSTOM_DELETER(PointSimpleListSearcher2);
}

PointSimpleListSearcher2&
PointSimpleListSearcher2::operator=(const PointSimpleListSearcher2& other) {
    set(other);
    return *this;
}

void PointSimpleListSearcher2::set(const PointSimpleListSearcher2& other) {
    _points = other._points;
}

void PointSimpleListSearcher2::serialize(
    std::vector<uint8_t>* buffer) const {
    flatbuffers::FlatBufferBuilder builder(1024);

    // Copy points
    std::vector<fbs::Vector2D> points;
    for (const auto& pt : _points) {
        points.push_back(jetToFbs(pt));
    }

    auto fbsPoints
        = builder.CreateVectorOfStructs(points.data(), points.size());

    // Copy the searcher
    auto fbsSearcher = fbs::CreatePointSimpleListSearcher2(builder, fbsPoints);

    builder.Finish(fbsSearcher);

    uint8_t *buf = builder.GetBufferPointer();
    size_t size = builder.GetSize();

    buffer->resize(size);
    memcpy(buffer->data(), buf, size);
}

void PointSimpleListSearcher2::deserialize(
    const std::vector<uint8_t>& buffer) {
    auto fbsSearcher = fbs::GetPointSimpleListSearcher2(buffer.data());

    // Copy points
    auto fbsPoints = fbsSearcher->points();
    _points.resize(fbsPoints->size());
    for (uint32_t i = 0; i < fbsPoints->size(); ++i) {
        _points[i] = fbsToJet(*fbsPoints->Get(i));
    }
}

PointSimpleListSearcher2
PointSimpleListSearcher2::Builder::build() const {
    return PointSimpleListSearcher2();
}

PointSimpleListSearcher2Ptr
PointSimpleListSearcher2::Builder::makeShared() const {
    return std::shared_ptr<PointSimpleListSearcher2>(
        new PointSimpleListSearcher2(),
        [] (PointSimpleListSearcher2* obj) {
            delete obj;
        });
}

PointNeighborSearcher2Ptr
PointSimpleListSearcher2::Builder::buildPointNeighborSearcher() const {
    return makeShared();
}
