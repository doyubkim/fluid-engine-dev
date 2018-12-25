// Copyright (c) 2018 Doyub Kim
//
// I am making my contributions/submissions to this project solely in my
// personal capacity and am not conveying any rights to any intellectual
// property of any third parties.

#ifdef _MSC_VER
#pragma warning(disable : 4244)
#endif

#include <common.h>

#include <fbs_helpers.h>
#include <generated/point_kdtree_searcher2_generated.h>
#include <generated/point_kdtree_searcher3_generated.h>

#include <jet/bounding_box.h>
#include <jet/point_kdtree_searcher.h>

#include <numeric>

namespace jet {

template <size_t N>
PointKdTreeSearcher<N>::PointKdTreeSearcher() {}

template <size_t N>
PointKdTreeSearcher<N>::PointKdTreeSearcher(const PointKdTreeSearcher &other) {
    set(other);
}

template <size_t N>
void PointKdTreeSearcher<N>::build(
    const ConstArrayView1<Vector<double, N>> &points, double maxSearchRadius) {
    UNUSED_VARIABLE(maxSearchRadius);

    _tree.build(points);
}

template <size_t N>
void PointKdTreeSearcher<N>::forEachNearbyPoint(
    const Vector<double, N> &origin, double radius,
    const ForEachNearbyPointFunc &callback) const {
    _tree.forEachNearbyPoint(origin, radius, callback);
}

template <size_t N>
bool PointKdTreeSearcher<N>::hasNearbyPoint(const Vector<double, N> &origin,
                                            double radius) const {
    return _tree.hasNearbyPoint(origin, radius);
}

template <size_t N>
std::shared_ptr<PointNeighborSearcher<N>> PointKdTreeSearcher<N>::clone()
    const {
    return CLONE_W_CUSTOM_DELETER(PointKdTreeSearcher);
}

template <size_t N>
PointKdTreeSearcher<N> &PointKdTreeSearcher<N>::operator=(
    const PointKdTreeSearcher &other) {
    set(other);
    return *this;
}

template <size_t N>
void PointKdTreeSearcher<N>::set(const PointKdTreeSearcher &other) {
    _tree = other._tree;
}

template <size_t N>
void PointKdTreeSearcher<N>::serialize(std::vector<uint8_t> *buffer) const {
    serialize(*this, buffer);
}

template <size_t N>
void PointKdTreeSearcher<N>::deserialize(const std::vector<uint8_t> &buffer) {
    deserialize(buffer, *this);
}

template <size_t N>
template <size_t M>
std::enable_if_t<M == 2, void> PointKdTreeSearcher<N>::serialize(
    const PointKdTreeSearcher<2> &searcher, std::vector<uint8_t> *buffer) {
    flatbuffers::FlatBufferBuilder builder(1024);

    // Copy points
    std::vector<fbs::Vector2D> points;
    for (const auto &iter : searcher._tree) {
        points.push_back(jetToFbs(iter));
    }

    auto fbsPoints =
        builder.CreateVectorOfStructs(points.data(), points.size());

    // Copy nodes
    std::vector<fbs::PointKdTreeSearcherNode2> nodes;
    for (auto iter = searcher._tree.beginNode();
         iter != searcher._tree.endNode(); ++iter) {
        nodes.emplace_back(iter->flags, iter->child, iter->item);
    }

    auto fbsNodes = builder.CreateVectorOfStructs(nodes);

    // Copy the searcher
    auto fbsSearcher =
        fbs::CreatePointKdTreeSearcher2(builder, fbsPoints, fbsNodes);

    // Finish
    builder.Finish(fbsSearcher);

    uint8_t *buf = builder.GetBufferPointer();
    size_t size = builder.GetSize();

    buffer->resize(size);
    memcpy(buffer->data(), buf, size);
}

template <size_t N>
template <size_t M>
std::enable_if_t<M == 3, void> PointKdTreeSearcher<N>::serialize(
    const PointKdTreeSearcher<3> &searcher, std::vector<uint8_t> *buffer) {
    flatbuffers::FlatBufferBuilder builder(1024);

    // Copy points
    std::vector<fbs::Vector3D> points;
    for (const auto &iter : searcher._tree) {
        points.push_back(jetToFbs(iter));
    }

    auto fbsPoints =
        builder.CreateVectorOfStructs(points.data(), points.size());

    // Copy nodes
    std::vector<fbs::PointKdTreeSearcherNode3> nodes;
    for (auto iter = searcher._tree.beginNode();
         iter != searcher._tree.endNode(); ++iter) {
        nodes.emplace_back(iter->flags, iter->child, iter->item);
    }

    auto fbsNodes = builder.CreateVectorOfStructs(nodes);

    // Copy the searcher
    auto fbsSearcher =
        fbs::CreatePointKdTreeSearcher3(builder, fbsPoints, fbsNodes);

    // Finish
    builder.Finish(fbsSearcher);

    uint8_t *buf = builder.GetBufferPointer();
    size_t size = builder.GetSize();

    buffer->resize(size);
    memcpy(buffer->data(), buf, size);
}

template <size_t N>
template <size_t M>
std::enable_if_t<M == 2, void> PointKdTreeSearcher<N>::deserialize(
    const std::vector<uint8_t> &buffer, PointKdTreeSearcher<2> &searcher) {
    auto fbsSearcher = fbs::GetPointKdTreeSearcher2(buffer.data());

    auto fbsPoints = fbsSearcher->points();
    auto fbsNodes = fbsSearcher->nodes();

    searcher._tree.reserve(fbsPoints->size(), fbsNodes->size());

    // Copy points
    auto pointsIter = searcher._tree.begin();
    for (uint32_t i = 0; i < fbsPoints->size(); ++i) {
        pointsIter[i] = fbsToJet(*fbsPoints->Get(i));
    }

    // Copy nodes
    auto nodesIter = searcher._tree.beginNode();
    for (uint32_t i = 0; i < fbsNodes->size(); ++i) {
        const auto fbsNode = fbsNodes->Get(i);
        nodesIter[i].flags = fbsNode->flags();
        nodesIter[i].child = fbsNode->child();
        nodesIter[i].item = fbsNode->item();
        nodesIter[i].point = pointsIter[fbsNode->item()];
    }
}

template <size_t N>
template <size_t M>
std::enable_if_t<M == 3, void> PointKdTreeSearcher<N>::deserialize(
    const std::vector<uint8_t> &buffer, PointKdTreeSearcher<3> &searcher) {
    auto fbsSearcher = fbs::GetPointKdTreeSearcher3(buffer.data());

    auto fbsPoints = fbsSearcher->points();
    auto fbsNodes = fbsSearcher->nodes();

    searcher._tree.reserve(fbsPoints->size(), fbsNodes->size());

    // Copy points
    auto pointsIter = searcher._tree.begin();
    for (uint32_t i = 0; i < fbsPoints->size(); ++i) {
        pointsIter[i] = fbsToJet(*fbsPoints->Get(i));
    }

    // Copy nodes
    auto nodesIter = searcher._tree.beginNode();
    for (uint32_t i = 0; i < fbsNodes->size(); ++i) {
        const auto fbsNode = fbsNodes->Get(i);
        nodesIter[i].flags = fbsNode->flags();
        nodesIter[i].child = fbsNode->child();
        nodesIter[i].item = fbsNode->item();
        nodesIter[i].point = pointsIter[fbsNode->item()];
    }
}

template <size_t N>
typename PointKdTreeSearcher<N>::Builder PointKdTreeSearcher<N>::builder() {
    return Builder{};
}

//

template <size_t N>
PointKdTreeSearcher<N> PointKdTreeSearcher<N>::Builder::build() const {
    return PointKdTreeSearcher{};
}

template <size_t N>
std::shared_ptr<PointKdTreeSearcher<N>>
PointKdTreeSearcher<N>::Builder::makeShared() const {
    return std::shared_ptr<PointKdTreeSearcher>(
        new PointKdTreeSearcher, [](PointKdTreeSearcher *obj) { delete obj; });
}

template <size_t N>
std::shared_ptr<PointNeighborSearcher<N>>
PointKdTreeSearcher<N>::Builder::buildPointNeighborSearcher() const {
    return makeShared();
}

template class PointKdTreeSearcher<2>;

template class PointKdTreeSearcher<3>;

}  // namespace jet
