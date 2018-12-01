// Copyright (c) 2018 Doyub Kim
//
// I am making my contributions/submissions to this project solely in my
// personal capacity and am not conveying any rights to any intellectual
// property of any third parties.

#include <pch.h>

#include <fbs_helpers.h>
#include <generated/point_simple_list_searcher2_generated.h>
#include <generated/point_simple_list_searcher3_generated.h>

#include <jet/point_simple_list_searcher.h>

namespace jet {

template <size_t N>
PointSimpleListSearcher<N>::PointSimpleListSearcher() {}

template <size_t N>
PointSimpleListSearcher<N>::PointSimpleListSearcher(
    const PointSimpleListSearcher &other) {
    set(other);
}

template <size_t N>
void PointSimpleListSearcher<N>::build(
    const ConstArrayView1<Vector<double, N>> &points, double maxSearchRadius) {
    UNUSED_VARIABLE(maxSearchRadius);

    _points.resize(points.length());
    std::copy(points.data(), points.data() + points.length(), _points.begin());
}

template <size_t N>
void PointSimpleListSearcher<N>::forEachNearbyPoint(
    const Vector<double, N> &origin, double radius,
    const ForEachNearbyPointFunc &callback) const {
    double radiusSquared = radius * radius;
    for (size_t i = 0; i < _points.length(); ++i) {
        Vector<double, N> r = _points[i] - origin;
        double distanceSquared = r.dot(r);
        if (distanceSquared <= radiusSquared) {
            callback(i, _points[i]);
        }
    }
}

template <size_t N>
bool PointSimpleListSearcher<N>::hasNearbyPoint(const Vector<double, N> &origin,
                                                double radius) const {
    double radiusSquared = radius * radius;
    for (size_t i = 0; i < _points.length(); ++i) {
        Vector<double, N> r = _points[i] - origin;
        double distanceSquared = r.dot(r);
        if (distanceSquared <= radiusSquared) {
            return true;
        }
    }

    return false;
}

template <size_t N>
std::shared_ptr<PointNeighborSearcher<N>> PointSimpleListSearcher<N>::clone()
    const {
    return CLONE_W_CUSTOM_DELETER(PointSimpleListSearcher);
}

template <size_t N>
PointSimpleListSearcher<N> &PointSimpleListSearcher<N>::operator=(
    const PointSimpleListSearcher &other) {
    set(other);
    return *this;
}

template <size_t N>
void PointSimpleListSearcher<N>::set(const PointSimpleListSearcher &other) {
    _points = other._points;
}

template <size_t N>
void PointSimpleListSearcher<N>::serialize(std::vector<uint8_t> *buffer) const {
    serialize(*this, buffer);
}

template <size_t N>
void PointSimpleListSearcher<N>::deserialize(
    const std::vector<uint8_t> &buffer) {
    deserialize(buffer, *this);
}

template <size_t N>
typename PointSimpleListSearcher<N>::Builder
PointSimpleListSearcher<N>::builder() {
    return Builder();
}

template <size_t N>
template <size_t M>
std::enable_if_t<M == 2, void> PointSimpleListSearcher<N>::serialize(
    const PointSimpleListSearcher<2> &searcher, std::vector<uint8_t> *buffer) {
    flatbuffers::FlatBufferBuilder builder(1024);

    // Copy points
    Array1<fbs::Vector2D> points;
    for (const auto &pt : searcher._points) {
        points.append(jetToFbs(pt));
    }

    auto fbsPoints =
        builder.CreateVectorOfStructs(points.data(), points.length());

    // Copy the searcher
    auto fbsSearcher = fbs::CreatePointSimpleListSearcher2(builder, fbsPoints);

    builder.Finish(fbsSearcher);

    uint8_t *buf = builder.GetBufferPointer();
    size_t size = builder.GetSize();

    buffer->resize(size);
    memcpy(buffer->data(), buf, size);
}

template <size_t N>
template <size_t M>
std::enable_if_t<M == 3, void> PointSimpleListSearcher<N>::serialize(
    const PointSimpleListSearcher<3> &searcher, std::vector<uint8_t> *buffer) {
    flatbuffers::FlatBufferBuilder builder(1024);

    // Copy points
    std::vector<fbs::Vector3D> points;
    for (const auto &pt : searcher._points) {
        points.push_back(jetToFbs(pt));
    }

    auto fbsPoints =
        builder.CreateVectorOfStructs(points.data(), points.size());

    // Copy the searcher
    auto fbsSearcher = fbs::CreatePointSimpleListSearcher3(builder, fbsPoints);

    builder.Finish(fbsSearcher);

    uint8_t *buf = builder.GetBufferPointer();
    size_t size = builder.GetSize();

    buffer->resize(size);
    memcpy(buffer->data(), buf, size);
}

template <size_t N>
template <size_t M>
std::enable_if_t<M == 2, void> PointSimpleListSearcher<N>::deserialize(
    const std::vector<uint8_t> &buffer, PointSimpleListSearcher<2> &searcher) {
    auto fbsSearcher = fbs::GetPointSimpleListSearcher2(buffer.data());

    // Copy points
    auto fbsPoints = fbsSearcher->points();
    searcher._points.resize(fbsPoints->size());
    for (uint32_t i = 0; i < fbsPoints->size(); ++i) {
        searcher._points[i] = fbsToJet(*fbsPoints->Get(i));
    }
}

template <size_t N>
template <size_t M>
std::enable_if_t<M == 3, void> PointSimpleListSearcher<N>::deserialize(
    const std::vector<uint8_t> &buffer, PointSimpleListSearcher<3> &searcher) {
    auto fbsSearcher = fbs::GetPointSimpleListSearcher3(buffer.data());

    // Copy points
    auto fbsPoints = fbsSearcher->points();
    searcher._points.resize(fbsPoints->size());
    for (uint32_t i = 0; i < fbsPoints->size(); ++i) {
        searcher._points[i] = fbsToJet(*fbsPoints->Get(i));
    }
}

template <size_t N>
PointSimpleListSearcher<N> PointSimpleListSearcher<N>::Builder::build() const {
    return PointSimpleListSearcher();
}

template <size_t N>
std::shared_ptr<PointSimpleListSearcher<N>>
PointSimpleListSearcher<N>::Builder::makeShared() const {
    return std::shared_ptr<PointSimpleListSearcher>(
        new PointSimpleListSearcher(),
        [](PointSimpleListSearcher *obj) { delete obj; });
}

template <size_t N>
std::shared_ptr<PointNeighborSearcher<N>>
PointSimpleListSearcher<N>::Builder::buildPointNeighborSearcher() const {
    return makeShared();
}

template class PointSimpleListSearcher<2>;

template class PointSimpleListSearcher<3>;

}  // namespace jet
