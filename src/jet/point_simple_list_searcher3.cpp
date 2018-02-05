// Copyright (c) 2018 Doyub Kim
//
// I am making my contributions/submissions to this project solely in my
// personal capacity and am not conveying any rights to any intellectual
// property of any third parties.

#include <pch.h>
#include <fbs_helpers.h>
#include <generated/point_simple_list_searcher3_generated.h>

#include <jet/point_simple_list_searcher3.h>

#include <algorithm>
#include <vector>

using namespace jet;

PointSimpleListSearcher3::PointSimpleListSearcher3() {
}

PointSimpleListSearcher3::PointSimpleListSearcher3(
    const PointSimpleListSearcher3& other) {
    set(other);
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

PointNeighborSearcher3Ptr PointSimpleListSearcher3::clone() const {
    return CLONE_W_CUSTOM_DELETER(PointSimpleListSearcher3);
}

PointSimpleListSearcher3&
PointSimpleListSearcher3::operator=(const PointSimpleListSearcher3& other) {
    set(other);
    return *this;
}

void PointSimpleListSearcher3::set(const PointSimpleListSearcher3& other) {
    _points = other._points;
}

void PointSimpleListSearcher3::serialize(
    std::vector<uint8_t>* buffer) const {
    flatbuffers::FlatBufferBuilder builder(1024);

    // Copy points
    std::vector<fbs::Vector3D> points;
    for (const auto& pt : _points) {
        points.push_back(jetToFbs(pt));
    }

    auto fbsPoints
        = builder.CreateVectorOfStructs(points.data(), points.size());

    // Copy the searcher
    auto fbsSearcher = fbs::CreatePointSimpleListSearcher3(builder, fbsPoints);

    builder.Finish(fbsSearcher);

    uint8_t *buf = builder.GetBufferPointer();
    size_t size = builder.GetSize();

    buffer->resize(size);
    memcpy(buffer->data(), buf, size);
}

void PointSimpleListSearcher3::deserialize(
    const std::vector<uint8_t>& buffer) {
    auto fbsSearcher = fbs::GetPointSimpleListSearcher3(buffer.data());

    // Copy points
    auto fbsPoints = fbsSearcher->points();
    _points.resize(fbsPoints->size());
    for (uint32_t i = 0; i < fbsPoints->size(); ++i) {
        _points[i] = fbsToJet(*fbsPoints->Get(i));
    }
}

PointSimpleListSearcher3
PointSimpleListSearcher3::Builder::build() const {
    return PointSimpleListSearcher3();
}

PointSimpleListSearcher3Ptr
PointSimpleListSearcher3::Builder::makeShared() const {
    return std::shared_ptr<PointSimpleListSearcher3>(
        new PointSimpleListSearcher3(),
        [] (PointSimpleListSearcher3* obj) {
            delete obj;
        });
}

PointNeighborSearcher3Ptr
PointSimpleListSearcher3::Builder::buildPointNeighborSearcher() const {
    return makeShared();
}
