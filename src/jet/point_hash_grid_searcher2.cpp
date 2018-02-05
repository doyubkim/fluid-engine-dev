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
#include <generated/point_hash_grid_searcher2_generated.h>

#include <jet/array1.h>
#include <jet/point_hash_grid_searcher2.h>

#include <algorithm>
#include <vector>

using namespace jet;

PointHashGridSearcher2::PointHashGridSearcher2(
    const Size2& resolution,
    double gridSpacing) :
    PointHashGridSearcher2(resolution.x, resolution.y, gridSpacing) {
}

PointHashGridSearcher2::PointHashGridSearcher2(
    size_t resolutionX,
    size_t resolutionY,
    double gridSpacing) :
    _gridSpacing(gridSpacing) {
    _resolution.x = std::max(static_cast<ssize_t>(resolutionX), kOneSSize);
    _resolution.y = std::max(static_cast<ssize_t>(resolutionY), kOneSSize);
}

PointHashGridSearcher2::PointHashGridSearcher2(
    const PointHashGridSearcher2& other) {
    set(other);
}

void PointHashGridSearcher2::build(
    const ConstArrayAccessor1<Vector2D>& points) {
    _buckets.clear();
    _points.clear();

    // Allocate memory chuncks
    _buckets.resize(_resolution.x * _resolution.y);
    _points.resize(points.size());

    if (points.size() == 0) {
        return;
    }

    // Put points into buckets
    for (size_t i = 0; i < points.size(); ++i) {
        _points[i] = points[i];
        size_t key = getHashKeyFromPosition(points[i]);
        _buckets[key].push_back(i);
    }
}

void PointHashGridSearcher2::forEachNearbyPoint(
    const Vector2D& origin,
    double radius,
    const ForEachNearbyPointFunc& callback) const {
    if (_buckets.empty()) {
        return;
    }

    size_t nearbyKeys[4];
    getNearbyKeys(origin, nearbyKeys);

    const double queryRadiusSquared = radius * radius;

    for (int i = 0; i < 4; i++) {
        const auto& bucket = _buckets[nearbyKeys[i]];
        size_t numberOfPointsInBucket = bucket.size();

        for (size_t j = 0; j < numberOfPointsInBucket; ++j) {
            size_t pointIndex = bucket[j];
            double rSquared = (_points[pointIndex] - origin).lengthSquared();
            if (rSquared <= queryRadiusSquared) {
                callback(pointIndex, _points[pointIndex]);
            }
        }
    }
}

bool PointHashGridSearcher2::hasNearbyPoint(
    const Vector2D& origin,
    double radius) const {
    if (_buckets.empty()) {
        return false;
    }

    size_t nearbyKeys[4];
    getNearbyKeys(origin, nearbyKeys);

    const double queryRadiusSquared = radius * radius;

    for (int i = 0; i < 4; i++) {
        const auto& bucket = _buckets[nearbyKeys[i]];
        size_t numberOfPointsInBucket = bucket.size();

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

void PointHashGridSearcher2::add(const Vector2D& point) {
    if (_buckets.empty()) {
        Array1<Vector2D> arr = {point};
        build(arr);
    } else {
        size_t i = _points.size();
        _points.push_back(point);
        size_t key = getHashKeyFromPosition(point);
        _buckets[key].push_back(i);
    }
}

const std::vector<std::vector<size_t>>&
PointHashGridSearcher2::buckets() const {
    return _buckets;
}

Point2I PointHashGridSearcher2::getBucketIndex(const Vector2D& position) const {
    Point2I bucketIndex;
    bucketIndex.x = static_cast<ssize_t>(
        std::floor(position.x / _gridSpacing));
    bucketIndex.y = static_cast<ssize_t>(
        std::floor(position.y / _gridSpacing));
    return bucketIndex;
}

size_t PointHashGridSearcher2::getHashKeyFromPosition(
    const Vector2D& position) const {
    Point2I bucketIndex = getBucketIndex(position);

    return getHashKeyFromBucketIndex(bucketIndex);
}

size_t PointHashGridSearcher2::getHashKeyFromBucketIndex(
    const Point2I& bucketIndex) const {
    Point2I wrappedIndex = bucketIndex;
    wrappedIndex.x = bucketIndex.x % _resolution.x;
    wrappedIndex.y = bucketIndex.y % _resolution.y;
    if (wrappedIndex.x < 0) {
        wrappedIndex.x += _resolution.x;
    }
    if (wrappedIndex.y < 0) {
        wrappedIndex.y += _resolution.y;
    }
    return static_cast<size_t>(wrappedIndex.y*_resolution.x + wrappedIndex.x);
}

void PointHashGridSearcher2::getNearbyKeys(
    const Vector2D& position,
    size_t* nearbyKeys) const {
    Point2I originIndex = getBucketIndex(position), nearbyBucketIndices[4];

    for (int i = 0; i < 4; i++) {
        nearbyBucketIndices[i] = originIndex;
    }

    if ((originIndex.x + 0.5f) * _gridSpacing <= position.x) {
        nearbyBucketIndices[2].x += 1;
        nearbyBucketIndices[3].x += 1;
    } else {
        nearbyBucketIndices[2].x -= 1;
        nearbyBucketIndices[3].x -= 1;
    }

    if ((originIndex.y + 0.5f) * _gridSpacing <= position.y) {
        nearbyBucketIndices[1].y += 1;
        nearbyBucketIndices[3].y += 1;
    } else {
        nearbyBucketIndices[1].y -= 1;
        nearbyBucketIndices[3].y -= 1;
    }

    for (int i = 0; i < 4; i++) {
        nearbyKeys[i] = getHashKeyFromBucketIndex(nearbyBucketIndices[i]);
    }
}

PointNeighborSearcher2Ptr PointHashGridSearcher2::clone() const {
    return CLONE_W_CUSTOM_DELETER(PointHashGridSearcher2);
}

PointHashGridSearcher2&
PointHashGridSearcher2::operator=(const PointHashGridSearcher2& other) {
    set(other);
    return *this;
}

void PointHashGridSearcher2::set(const PointHashGridSearcher2& other) {
    _gridSpacing = other._gridSpacing;
    _resolution = other._resolution;
    _points = other._points;
    _buckets = other._buckets;
}

void PointHashGridSearcher2::serialize(std::vector<uint8_t>* buffer) const {
    flatbuffers::FlatBufferBuilder builder(1024);

    // Copy simple data
    auto fbsResolution = fbs::Size2(_resolution.x, _resolution.y);

    // Copy points
    std::vector<fbs::Vector2D> points;
    for (const auto& pt : _points) {
        points.push_back(jetToFbs(pt));
    }

    auto fbsPoints
        = builder.CreateVectorOfStructs(points.data(), points.size());

    // Copy buckets
    std::vector<flatbuffers::Offset<fbs::PointHashGridSearcherBucket2>> buckets;
    for (const auto& bucket : _buckets) {
        std::vector<uint64_t> bucket64(bucket.begin(), bucket.end());
        flatbuffers::Offset<fbs::PointHashGridSearcherBucket2> fbsBucket
            = fbs::CreatePointHashGridSearcherBucket2(
                builder,
                builder.CreateVector(bucket64.data(), bucket64.size()));
        buckets.push_back(fbsBucket);
    }

    auto fbsBuckets = builder.CreateVector(buckets);

    // Copy the searcher
    auto fbsSearcher = fbs::CreatePointHashGridSearcher2(
        builder, _gridSpacing, &fbsResolution, fbsPoints, fbsBuckets);

    builder.Finish(fbsSearcher);

    uint8_t *buf = builder.GetBufferPointer();
    size_t size = builder.GetSize();

    buffer->resize(size);
    memcpy(buffer->data(), buf, size);
}

void PointHashGridSearcher2::deserialize(const std::vector<uint8_t>& buffer) {
    auto fbsSearcher = fbs::GetPointHashGridSearcher2(buffer.data());

    // Copy simple data
    auto res = fbsToJet(*fbsSearcher->resolution());
    _resolution.set({res.x, res.y});
    _gridSpacing = fbsSearcher->gridSpacing();

    // Copy points
    auto fbsPoints = fbsSearcher->points();
    _points.resize(fbsPoints->size());
    for (uint32_t i = 0; i < fbsPoints->size(); ++i) {
        _points[i] = fbsToJet(*fbsPoints->Get(i));
    }

    // Copy buckets
    auto fbsBuckets = fbsSearcher->buckets();
    _buckets.resize(fbsBuckets->size());
    for (uint32_t i = 0; i < fbsBuckets->size(); ++i) {
        auto fbsBucket = fbsBuckets->Get(i);
        _buckets[i].resize(fbsBucket->data()->size());
        std::transform(
            fbsBucket->data()->begin(),
            fbsBucket->data()->end(),
            _buckets[i].begin(),
            [] (uint64_t val) {
                return static_cast<size_t>(val);
            });
    }
}

PointHashGridSearcher2::Builder PointHashGridSearcher2::builder() {
    return Builder();
}


PointHashGridSearcher2::Builder&
PointHashGridSearcher2::Builder::withResolution(const Size2& resolution) {
    _resolution = resolution;
    return *this;
}

PointHashGridSearcher2::Builder&
PointHashGridSearcher2::Builder::withGridSpacing(double gridSpacing) {
    _gridSpacing = gridSpacing;
    return *this;
}

PointHashGridSearcher2
PointHashGridSearcher2::Builder::build() const {
    return PointHashGridSearcher2(_resolution, _gridSpacing);
}

PointHashGridSearcher2Ptr
PointHashGridSearcher2::Builder::makeShared() const {
    return std::shared_ptr<PointHashGridSearcher2>(
        new PointHashGridSearcher2(_resolution, _gridSpacing),
        [] (PointHashGridSearcher2* obj) {
            delete obj;
        });
}

PointNeighborSearcher2Ptr
PointHashGridSearcher2::Builder::buildPointNeighborSearcher() const {
    return makeShared();
}
