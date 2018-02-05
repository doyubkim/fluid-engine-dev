// Copyright (c) 2018 Doyub Kim
//
// I am making my contributions/submissions to this project solely in my
// personal capacity and am not conveying any rights to any intellectual
// property of any third parties.

#ifdef _MSC_VER
#pragma warning(disable: 4244)
#endif

#include <pch.h>

#include <fbs_helpers.h>
#include <generated/point_kdtree_searcher3_generated.h>

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
    flatbuffers::FlatBufferBuilder builder(1024);

    // Copy points
    std::vector<fbs::Vector3D> points;
    for (const auto& iter : _tree) {
        points.push_back(jetToFbs(iter));
    }

    auto fbsPoints =
            builder.CreateVectorOfStructs(points.data(), points.size());

    // Copy nodes
    std::vector<fbs::PointKdTreeSearcherNode3> nodes;
    for (auto iter = _tree.beginNode(); iter != _tree.endNode(); ++iter) {
        nodes.emplace_back(iter->flags, iter->child, iter->item);
    }

    auto fbsNodes = builder.CreateVectorOfStructs(nodes);

    // Copy the searcher
    auto fbsSearcher =
            fbs::CreatePointKdTreeSearcher3(builder, fbsPoints, fbsNodes);

    // Finish
    builder.Finish(fbsSearcher);

    uint8_t* buf = builder.GetBufferPointer();
    size_t size = builder.GetSize();

    buffer->resize(size);
    memcpy(buffer->data(), buf, size);
}

void PointKdTreeSearcher3::deserialize(const std::vector<uint8_t>& buffer) {
    auto fbsSearcher = fbs::GetPointKdTreeSearcher3(buffer.data());

    auto fbsPoints = fbsSearcher->points();
    auto fbsNodes = fbsSearcher->nodes();

    _tree.reserve(fbsPoints->size(), fbsNodes->size());

    // Copy points
    auto pointsIter = _tree.begin();
    for (uint32_t i = 0; i < fbsPoints->size(); ++i) {
        pointsIter[i] = fbsToJet(*fbsPoints->Get(i));
    }

    // Copy nodes
    auto nodesIter = _tree.beginNode();
    for (uint32_t i = 0; i < fbsNodes->size(); ++i) {
        const auto fbsNode = fbsNodes->Get(i);
        nodesIter[i].flags = fbsNode->flags();
        nodesIter[i].child = fbsNode->child();
        nodesIter[i].item = fbsNode->item();
        nodesIter[i].point = pointsIter[fbsNode->item()];
    }
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
