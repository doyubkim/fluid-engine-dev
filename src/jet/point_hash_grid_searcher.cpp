// Copyright (c) 2018 Doyub Kim
//
// I am making my contributions/submissions to this project solely in my
// personal capacity and am not conveying any rights to any intellectual
// property of any third parties.

#ifdef _MSC_VER
#pragma warning(disable : 4244)
#endif

#include <pch.h>

#include <fbs_helpers.h>
#include <generated/point_hash_grid_searcher2_generated.h>
#include <generated/point_hash_grid_searcher3_generated.h>

#include <jet/array.h>
#include <jet/point_hash_grid_searcher.h>

#include <algorithm>
#include <vector>

namespace jet {

inline size_t hashKey(const Vector<ssize_t, 2> &index,
                      const Vector<ssize_t, 2> &resolution) {
    return static_cast<size_t>(index.y * resolution.x + index.x);
}

inline size_t hashKey(const Vector<ssize_t, 3> &index,
                      const Vector<ssize_t, 3> &resolution) {
    return static_cast<size_t>(
        (index.z * resolution.y + index.y) * resolution.x + index.x);
}

template <size_t N>
PointHashGridSearcher<N>::PointHashGridSearcher(
    const Vector<size_t, N> &resolution, double gridSpacing) {
    _gridSpacing = gridSpacing;
    _resolution = max(resolution.template castTo<ssize_t>(),
                      Vector<ssize_t, N>::makeConstant(kOneSSize));
}

template <size_t N>
PointHashGridSearcher<N>::PointHashGridSearcher(
    const PointHashGridSearcher &other) {
    set(other);
}

template <size_t N>
void PointHashGridSearcher<N>::build(
    const ConstArrayView1<Vector<double, N>> &points) {
    _buckets.clear();
    _points.clear();

    // Allocate memory chunks
    _buckets.resize(product(_resolution, kOneSSize));
    _points.resize(points.length());

    if (points.length() == 0) {
        return;
    }

    // Put points into buckets
    for (size_t i = 0; i < points.length(); ++i) {
        _points[i] = points[i];
        size_t key = getHashKeyFromPosition(points[i]);
        _buckets[key].append(i);
    }
}

template <size_t N>
void PointHashGridSearcher<N>::forEachNearbyPoint(
    const Vector<double, N> &origin, double radius,
    const ForEachNearbyPointFunc &callback) const {
    if (_buckets.isEmpty()) {
        return;
    }

    constexpr int kNumKeys = 1 << N;
    size_t nearbyKeys[kNumKeys];
    getNearbyKeys(origin, nearbyKeys);

    const double queryRadiusSquared = radius * radius;

    for (int i = 0; i < kNumKeys; i++) {
        const auto &bucket = _buckets[nearbyKeys[i]];
        size_t numberOfPointsInBucket = bucket.length();

        for (size_t j = 0; j < numberOfPointsInBucket; ++j) {
            size_t pointIndex = bucket[j];
            double rSquared = (_points[pointIndex] - origin).lengthSquared();
            if (rSquared <= queryRadiusSquared) {
                callback(pointIndex, _points[pointIndex]);
            }
        }
    }
}

template <size_t N>
bool PointHashGridSearcher<N>::hasNearbyPoint(const Vector<double, N> &origin,
                                              double radius) const {
    if (_buckets.isEmpty()) {
        return false;
    }

    constexpr int kNumKeys = 1 << N;
    size_t nearbyKeys[kNumKeys];
    getNearbyKeys(origin, nearbyKeys);

    const double queryRadiusSquared = radius * radius;

    for (int i = 0; i < kNumKeys; i++) {
        const auto &bucket = _buckets[nearbyKeys[i]];
        size_t numberOfPointsInBucket = bucket.length();

        for (size_t j = 0; j < numberOfPointsInBucket; ++j) {
            size_t pointIndex = bucket[j];
            double rSquared = (_points[pointIndex] - origin).lengthSquared();
            if (rSquared <= queryRadiusSquared) {
                return true;
            }
        }
    }

    return false;
}

template <size_t N>
void PointHashGridSearcher<N>::add(const Vector<double, N> &point) {
    if (_buckets.isEmpty()) {
        Array1<Vector<double, N>> arr = {point};
        build(arr);
    } else {
        size_t i = _points.length();
        _points.append(point);
        size_t key = getHashKeyFromPosition(point);
        _buckets[key].append(i);
    }
}

template <size_t N>
const Array1<Array1<size_t>> &PointHashGridSearcher<N>::buckets() const {
    return _buckets;
}

template <size_t N>
Vector<ssize_t, N> PointHashGridSearcher<N>::getBucketIndex(
    const Vector<double, N> &position) const {
    Vector<ssize_t, N> bucketIndex;
    bucketIndex = floor(position / _gridSpacing).template castTo<ssize_t>();
    return bucketIndex;
}

template <size_t N>
size_t PointHashGridSearcher<N>::getHashKeyFromPosition(
    const Vector<double, N> &position) const {
    Vector<ssize_t, N> bucketIndex = getBucketIndex(position);

    return getHashKeyFromBucketIndex(bucketIndex);
}

template <size_t N>
size_t PointHashGridSearcher<N>::getHashKeyFromBucketIndex(
    const Vector<ssize_t, N> &bucketIndex) const {
    Vector<ssize_t, N> wrappedIndex = bucketIndex;
    for (size_t i = 0; i < N; ++i) {
        wrappedIndex[i] = bucketIndex[i] % _resolution[i];

        if (wrappedIndex[i] < 0) {
            wrappedIndex[i] += _resolution[i];
        }
    }

    return hashKey(wrappedIndex, _resolution);
}

template <size_t N>
void PointHashGridSearcher<N>::getNearbyKeys(const Vector<double, N> &position,
                                             size_t *nearbyKeys) const {
    constexpr int kNumKeys = 1 << N;

    Vector<ssize_t, N> originIndex = getBucketIndex(position);
    Vector<ssize_t, N> nearbyBucketIndices[kNumKeys];

    for (int i = 0; i < kNumKeys; i++) {
        nearbyBucketIndices[i] = originIndex;
    }

    for (size_t axis = 0; axis < N; axis++) {
        int offset =
            (originIndex[axis] + 0.5) * _gridSpacing <= position[axis] ? 1 : -1;
        for (int j = 0; j < kNumKeys; j++) {
            if (j & (kNumKeys >> axis)) {
                nearbyBucketIndices[j][axis] += offset;
            }
        }
    }

    for (int i = 0; i < kNumKeys; i++) {
        nearbyKeys[i] = getHashKeyFromBucketIndex(nearbyBucketIndices[i]);
    }
}

template <size_t N>
std::shared_ptr<PointNeighborSearcher<N>> PointHashGridSearcher<N>::clone()
    const {
    return CLONE_W_CUSTOM_DELETER(PointHashGridSearcher);
}

template <size_t N>
PointHashGridSearcher<N> &PointHashGridSearcher<N>::operator=(
    const PointHashGridSearcher &other) {
    set(other);
    return *this;
}

template <size_t N>
void PointHashGridSearcher<N>::set(const PointHashGridSearcher &other) {
    _gridSpacing = other._gridSpacing;
    _resolution = other._resolution;
    _points = other._points;
    _buckets = other._buckets;
}

template <size_t N>
void PointHashGridSearcher<N>::serialize(std::vector<uint8_t> *buffer) const {
    serialize(*this, buffer);
}

template <size_t N>
void PointHashGridSearcher<N>::deserialize(const std::vector<uint8_t> &buffer) {
    deserialize(buffer, *this);
}

template <size_t N>
template <size_t M>
std::enable_if_t<M == 2, void> PointHashGridSearcher<N>::serialize(
    const PointHashGridSearcher<2> &searcher, std::vector<uint8_t> *buffer) {
    flatbuffers::FlatBufferBuilder builder(1024);

    // Copy simple data
    auto fbsResolution =
        fbs::Vector2UZ(searcher._resolution.x, searcher._resolution.y);

    // Copy points
    std::vector<fbs::Vector2D> points;
    for (const auto &pt : searcher._points) {
        points.push_back(jetToFbs(pt));
    }

    auto fbsPoints =
        builder.CreateVectorOfStructs(points.data(), points.size());

    // Copy buckets
    std::vector<flatbuffers::Offset<fbs::PointHashGridSearcherBucket2>> buckets;
    for (const auto &bucket : searcher._buckets) {
        std::vector<uint64_t> bucket64(bucket.begin(), bucket.end());
        flatbuffers::Offset<fbs::PointHashGridSearcherBucket2> fbsBucket =
            fbs::CreatePointHashGridSearcherBucket2(
                builder,
                builder.CreateVector(bucket64.data(), bucket64.size()));
        buckets.push_back(fbsBucket);
    }

    auto fbsBuckets = builder.CreateVector(buckets);

    // Copy the searcher
    auto fbsSearcher = fbs::CreatePointHashGridSearcher2(
        builder, searcher._gridSpacing, &fbsResolution, fbsPoints, fbsBuckets);

    builder.Finish(fbsSearcher);

    uint8_t *buf = builder.GetBufferPointer();
    size_t size = builder.GetSize();

    buffer->resize(size);
    memcpy(buffer->data(), buf, size);
}

template <size_t N>
template <size_t M>
std::enable_if_t<M == 3, void> PointHashGridSearcher<N>::serialize(
    const PointHashGridSearcher<3> &searcher, std::vector<uint8_t> *buffer) {
    flatbuffers::FlatBufferBuilder builder(1024);

    // Copy simple data
    auto fbsResolution = fbs::Vector3UZ(
        searcher._resolution.x, searcher._resolution.y, searcher._resolution.z);

    // Copy points
    std::vector<fbs::Vector3D> points;
    for (const auto &pt : searcher._points) {
        points.push_back(jetToFbs(pt));
    }

    auto fbsPoints =
        builder.CreateVectorOfStructs(points.data(), points.size());

    // Copy buckets
    std::vector<flatbuffers::Offset<fbs::PointHashGridSearcherBucket3>> buckets;
    for (const auto &bucket : searcher._buckets) {
        std::vector<uint64_t> bucket64(bucket.begin(), bucket.end());
        flatbuffers::Offset<fbs::PointHashGridSearcherBucket3> fbsBucket =
            fbs::CreatePointHashGridSearcherBucket3(
                builder,
                builder.CreateVector(bucket64.data(), bucket64.size()));
        buckets.push_back(fbsBucket);
    }

    auto fbsBuckets = builder.CreateVector(buckets);

    // Copy the searcher
    auto fbsSearcher = fbs::CreatePointHashGridSearcher3(
        builder, searcher._gridSpacing, &fbsResolution, fbsPoints, fbsBuckets);

    builder.Finish(fbsSearcher);

    uint8_t *buf = builder.GetBufferPointer();
    size_t size = builder.GetSize();

    buffer->resize(size);
    memcpy(buffer->data(), buf, size);
}

template <size_t N>
template <size_t M>
std::enable_if_t<M == 2, void> PointHashGridSearcher<N>::deserialize(
    const std::vector<uint8_t> &buffer, PointHashGridSearcher<2> &searcher) {
    auto fbsSearcher = fbs::GetPointHashGridSearcher2(buffer.data());

    // Copy simple data
    auto res = fbsToJet(*fbsSearcher->resolution());
    searcher._resolution = {(ssize_t)res.x, (ssize_t)res.y};
    searcher._gridSpacing = fbsSearcher->gridSpacing();

    // Copy points
    auto fbsPoints = fbsSearcher->points();
    searcher._points.resize(fbsPoints->size());
    for (uint32_t i = 0; i < fbsPoints->size(); ++i) {
        searcher._points[i] = fbsToJet(*fbsPoints->Get(i));
    }

    // Copy buckets
    auto fbsBuckets = fbsSearcher->buckets();
    searcher._buckets.resize(fbsBuckets->size());
    for (uint32_t i = 0; i < fbsBuckets->size(); ++i) {
        auto fbsBucket = fbsBuckets->Get(i);
        searcher._buckets[i].resize(fbsBucket->data()->size());
        std::transform(fbsBucket->data()->begin(), fbsBucket->data()->end(),
                       searcher._buckets[i].begin(),
                       [](uint64_t val) { return static_cast<size_t>(val); });
    }
}

template <size_t N>
template <size_t M>
std::enable_if_t<M == 3, void> PointHashGridSearcher<N>::deserialize(
    const std::vector<uint8_t> &buffer, PointHashGridSearcher<3> &searcher) {
    auto fbsSearcher = fbs::GetPointHashGridSearcher3(buffer.data());

    // Copy simple data
    auto res = fbsToJet(*fbsSearcher->resolution());
    searcher._resolution = {(ssize_t)res.x, (ssize_t)res.y, (ssize_t)res.z};
    searcher._gridSpacing = fbsSearcher->gridSpacing();

    // Copy points
    auto fbsPoints = fbsSearcher->points();
    searcher._points.resize(fbsPoints->size());
    for (uint32_t i = 0; i < fbsPoints->size(); ++i) {
        searcher._points[i] = fbsToJet(*fbsPoints->Get(i));
    }

    // Copy buckets
    auto fbsBuckets = fbsSearcher->buckets();
    searcher._buckets.resize(fbsBuckets->size());
    for (uint32_t i = 0; i < fbsBuckets->size(); ++i) {
        auto fbsBucket = fbsBuckets->Get(i);
        searcher._buckets[i].resize(fbsBucket->data()->size());
        std::transform(fbsBucket->data()->begin(), fbsBucket->data()->end(),
                       searcher._buckets[i].begin(),
                       [](uint64_t val) { return static_cast<size_t>(val); });
    }
}

template <size_t N>
typename PointHashGridSearcher<N>::Builder PointHashGridSearcher<N>::builder() {
    return Builder();
}

template <size_t N>
typename PointHashGridSearcher<N>::Builder &
PointHashGridSearcher<N>::Builder::withResolution(
    const Vector<size_t, N> &resolution) {
    _resolution = resolution;
    return *this;
}

template <size_t N>
typename PointHashGridSearcher<N>::Builder &
PointHashGridSearcher<N>::Builder::withGridSpacing(double gridSpacing) {
    _gridSpacing = gridSpacing;
    return *this;
}

template <size_t N>
PointHashGridSearcher<N> PointHashGridSearcher<N>::Builder::build() const {
    return PointHashGridSearcher(_resolution, _gridSpacing);
}

template <size_t N>
std::shared_ptr<PointHashGridSearcher<N>>
PointHashGridSearcher<N>::Builder::makeShared() const {
    return std::shared_ptr<PointHashGridSearcher>(
        new PointHashGridSearcher(_resolution, _gridSpacing),
        [](PointHashGridSearcher *obj) { delete obj; });
}

template <size_t N>
std::shared_ptr<PointNeighborSearcher<N>>
PointHashGridSearcher<N>::Builder::buildPointNeighborSearcher() const {
    return makeShared();
}

template class PointHashGridSearcher<2>;

template class PointHashGridSearcher<3>;

}  // namespace jet
