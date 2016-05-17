// Copyright (c) 2016 Doyub Kim

#include <pch.h>

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
    if (wrappedIndex.x < 0) { wrappedIndex.x += _resolution.x; }
    if (wrappedIndex.y < 0) { wrappedIndex.y += _resolution.y; }
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
        nearbyBucketIndices[2].x += 1; nearbyBucketIndices[3].x += 1;
    } else {
        nearbyBucketIndices[2].x -= 1; nearbyBucketIndices[3].x -= 1;
    }

    if ((originIndex.y + 0.5f) * _gridSpacing <= position.y) {
        nearbyBucketIndices[1].y += 1; nearbyBucketIndices[3].y += 1;
    } else {
        nearbyBucketIndices[1].y -= 1; nearbyBucketIndices[3].y -= 1;
    }

    for (int i = 0; i < 4; i++) {
        nearbyKeys[i] = getHashKeyFromBucketIndex(nearbyBucketIndices[i]);
    }
}
