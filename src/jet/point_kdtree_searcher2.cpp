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
#include <generated/point_kdtree_searcher2_generated.h>

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
    flatbuffers::FlatBufferBuilder builder(1024);

    // Copy points
    std::vector<fbs::Vector2D> points;
    for (const auto& iter : _tree) {
        points.push_back(jetToFbs(iter));
    }

    auto fbsPoints =
        builder.CreateVectorOfStructs(points.data(), points.size());

    // Copy nodes
    std::vector<fbs::PointKdTreeSearcherNode2> nodes;
    for (auto iter = _tree.beginNode(); iter != _tree.endNode(); ++iter) {
        nodes.emplace_back(iter->flags, iter->child, iter->item);
    }

    auto fbsNodes = builder.CreateVectorOfStructs(nodes);

    // Copy the searcher
    auto fbsSearcher =
        fbs::CreatePointKdTreeSearcher2(builder, fbsPoints, fbsNodes);

    // Finish
    builder.Finish(fbsSearcher);

    uint8_t* buf = builder.GetBufferPointer();
    size_t size = builder.GetSize();

    buffer->resize(size);
    memcpy(buffer->data(), buf, size);
}

void PointKdTreeSearcher2::deserialize(const std::vector<uint8_t>& buffer) {
    auto fbsSearcher = fbs::GetPointKdTreeSearcher2(buffer.data());

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
