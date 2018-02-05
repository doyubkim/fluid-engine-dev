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
#include <generated/point_parallel_hash_grid_searcher3_generated.h>

#include <jet/constants.h>
#include <jet/parallel.h>
#include <jet/point_parallel_hash_grid_searcher3.h>

#include <algorithm>
#include <vector>

using namespace jet;

PointParallelHashGridSearcher3::PointParallelHashGridSearcher3(
    const Size3& resolution,
    double gridSpacing) :
    PointParallelHashGridSearcher3(
        resolution.x, resolution.y, resolution.z, gridSpacing) {
}

PointParallelHashGridSearcher3::PointParallelHashGridSearcher3(
    size_t resolutionX,
    size_t resolutionY,
    size_t resolutionZ,
    double gridSpacing) :
    _gridSpacing(gridSpacing) {
    _resolution.x = std::max(static_cast<ssize_t>(resolutionX), kOneSSize);
    _resolution.y = std::max(static_cast<ssize_t>(resolutionY), kOneSSize);
    _resolution.z = std::max(static_cast<ssize_t>(resolutionZ), kOneSSize);

    _startIndexTable.resize(
        _resolution.x * _resolution.y * _resolution.z, kMaxSize);
    _endIndexTable.resize(
        _resolution.x * _resolution.y * _resolution.z, kMaxSize);
}

PointParallelHashGridSearcher3::PointParallelHashGridSearcher3(
    const PointParallelHashGridSearcher3& other) {
    set(other);
}

void PointParallelHashGridSearcher3::build(
    const ConstArrayAccessor1<Vector3D>& points) {
    _points.clear();
    _keys.clear();
    _startIndexTable.clear();
    _endIndexTable.clear();
    _sortedIndices.clear();

    // Allocate memory chuncks
    size_t numberOfPoints = points.size();
    std::vector<size_t> tempKeys(numberOfPoints);
    _startIndexTable.resize(_resolution.x * _resolution.y * _resolution.z);
    _endIndexTable.resize(_resolution.x * _resolution.y * _resolution.z);
    parallelFill(_startIndexTable.begin(), _startIndexTable.end(), kMaxSize);
    parallelFill(_endIndexTable.begin(), _endIndexTable.end(), kMaxSize);
    _keys.resize(numberOfPoints);
    _sortedIndices.resize(numberOfPoints);
    _points.resize(numberOfPoints);

    if (numberOfPoints == 0) {
        return;
    }

    // Initialize indices array and generate hash key for each point
    parallelFor(
        kZeroSize,
        numberOfPoints,
        [&](size_t i) {
            _sortedIndices[i] = i;
            _points[i] = points[i];
            tempKeys[i] = getHashKeyFromPosition(points[i]);
        });

    // Sort indices based on hash key
    parallelSort(
        _sortedIndices.begin(),
        _sortedIndices.end(),
        [&tempKeys](size_t indexA, size_t indexB) {
            return tempKeys[indexA] < tempKeys[indexB];
        });

    // Re-order point and key arrays
    parallelFor(
        kZeroSize,
        numberOfPoints,
        [&](size_t i) {
            _points[i] = points[_sortedIndices[i]];
            _keys[i] = tempKeys[_sortedIndices[i]];
        });

    // Now _points and _keys are sorted by points' hash key values.
    // Let's fill in start/end index table with _keys.

    // Assume that _keys array looks like:
    // [5|8|8|10|10|10]
    // Then _startIndexTable and _endIndexTable should be like:
    // [.....|0|...|1|..|3|..]
    // [.....|1|...|3|..|6|..]
    //       ^5    ^8   ^10
    // So that _endIndexTable[i] - _startIndexTable[i] is the number points
    // in i-th table bucket.

    _startIndexTable[_keys[0]] = 0;
    _endIndexTable[_keys[numberOfPoints - 1]] = numberOfPoints;

    parallelFor(
        (size_t)1,
        numberOfPoints,
        [&](size_t i) {
            if (_keys[i] > _keys[i - 1]) {
                _startIndexTable[_keys[i]] = i;
                _endIndexTable[_keys[i - 1]] = i;
            }
        });

    size_t sumNumberOfPointsPerBucket = 0;
    size_t maxNumberOfPointsPerBucket = 0;
    size_t numberOfNonEmptyBucket = 0;
    for (size_t i = 0; i < _startIndexTable.size(); ++i) {
        if (_startIndexTable[i] != kMaxSize) {
            size_t numberOfPointsInBucket
                = _endIndexTable[i] - _startIndexTable[i];
            sumNumberOfPointsPerBucket += numberOfPointsInBucket;
            maxNumberOfPointsPerBucket =
                std::max(maxNumberOfPointsPerBucket, numberOfPointsInBucket);
            ++numberOfNonEmptyBucket;
        }
    }

    JET_INFO << "Average number of points per non-empty bucket: "
             << static_cast<float>(sumNumberOfPointsPerBucket)
                / static_cast<float>(numberOfNonEmptyBucket);
    JET_INFO << "Max number of points per bucket: "
             << maxNumberOfPointsPerBucket;
}

void PointParallelHashGridSearcher3::forEachNearbyPoint(
    const Vector3D& origin,
    double radius,
    const ForEachNearbyPointFunc& callback) const {
    size_t nearbyKeys[8];
    getNearbyKeys(origin, nearbyKeys);

    const double queryRadiusSquared = radius * radius;

    for (int i = 0; i < 8; i++) {
        size_t nearbyKey = nearbyKeys[i];
        size_t start = _startIndexTable[nearbyKey];
        size_t end = _endIndexTable[nearbyKey];

        // Empty bucket -- continue to next bucket
        if (start == kMaxSize) {
            continue;
        }

        for (size_t j = start; j < end; ++j) {
            Vector3D direction = _points[j] - origin;
            double distanceSquared = direction.lengthSquared();
            if (distanceSquared <= queryRadiusSquared) {
                double distance = 0.0;
                if (distanceSquared > 0) {
                    distance = std::sqrt(distanceSquared);
                    direction /= distance;
                }

                callback(_sortedIndices[j], _points[j]);
            }
        }
    }
}

bool PointParallelHashGridSearcher3::hasNearbyPoint(
    const Vector3D& origin,
    double radius) const {
    size_t nearbyKeys[8];
    getNearbyKeys(origin, nearbyKeys);

    const double queryRadiusSquared = radius * radius;

    for (int i = 0; i < 8; i++) {
        size_t nearbyKey = nearbyKeys[i];
        size_t start = _startIndexTable[nearbyKey];
        size_t end = _endIndexTable[nearbyKey];

        // Empty bucket -- continue to next bucket
        if (start == kMaxSize) {
            continue;
        }

        for (size_t j = start; j < end; ++j) {
            Vector3D direction = _points[j] - origin;
            double distanceSquared = direction.lengthSquared();
            if (distanceSquared <= queryRadiusSquared) {
                return true;
            }
        }
    }

    return false;
}

const std::vector<size_t>& PointParallelHashGridSearcher3::keys() const {
    return _keys;
}

const std::vector<size_t>&
PointParallelHashGridSearcher3::startIndexTable() const {
    return _startIndexTable;
}

const std::vector<size_t>&
PointParallelHashGridSearcher3::endIndexTable() const {
    return _endIndexTable;
}

const std::vector<size_t>&
PointParallelHashGridSearcher3::sortedIndices() const {
    return _sortedIndices;
}

Point3I PointParallelHashGridSearcher3::getBucketIndex(
    const Vector3D& position) const {
    Point3I bucketIndex;
    bucketIndex.x = static_cast<ssize_t>(
        std::floor(position.x / _gridSpacing));
    bucketIndex.y = static_cast<ssize_t>(
        std::floor(position.y / _gridSpacing));
    bucketIndex.z = static_cast<ssize_t>(
        std::floor(position.z / _gridSpacing));
    return bucketIndex;
}

size_t PointParallelHashGridSearcher3::getHashKeyFromPosition(
    const Vector3D& position) const {
    Point3I bucketIndex = getBucketIndex(position);

    return getHashKeyFromBucketIndex(bucketIndex);
}

size_t PointParallelHashGridSearcher3::getHashKeyFromBucketIndex(
    const Point3I& bucketIndex) const {
    Point3I wrappedIndex = bucketIndex;
    wrappedIndex.x = bucketIndex.x % _resolution.x;
    wrappedIndex.y = bucketIndex.y % _resolution.y;
    wrappedIndex.z = bucketIndex.z % _resolution.z;
    if (wrappedIndex.x < 0) {
        wrappedIndex.x += _resolution.x;
    }
    if (wrappedIndex.y < 0) {
        wrappedIndex.y += _resolution.y;
    }
    if (wrappedIndex.z < 0) {
        wrappedIndex.z += _resolution.z;
    }
    return static_cast<size_t>(
        (wrappedIndex.z * _resolution.y + wrappedIndex.y) * _resolution.x
        + wrappedIndex.x);
}

void PointParallelHashGridSearcher3::getNearbyKeys(
    const Vector3D& position,
    size_t* nearbyKeys) const {
    Point3I originIndex = getBucketIndex(position), nearbyBucketIndices[8];

    for (int i = 0; i < 8; i++) {
        nearbyBucketIndices[i] = originIndex;
    }

    if ((originIndex.x + 0.5f) * _gridSpacing <= position.x) {
        nearbyBucketIndices[4].x += 1;
        nearbyBucketIndices[5].x += 1;
        nearbyBucketIndices[6].x += 1;
        nearbyBucketIndices[7].x += 1;
    } else {
        nearbyBucketIndices[4].x -= 1;
        nearbyBucketIndices[5].x -= 1;
        nearbyBucketIndices[6].x -= 1;
        nearbyBucketIndices[7].x -= 1;
    }

    if ((originIndex.y + 0.5f) * _gridSpacing <= position.y) {
        nearbyBucketIndices[2].y += 1;
        nearbyBucketIndices[3].y += 1;
        nearbyBucketIndices[6].y += 1;
        nearbyBucketIndices[7].y += 1;
    } else {
        nearbyBucketIndices[2].y -= 1;
        nearbyBucketIndices[3].y -= 1;
        nearbyBucketIndices[6].y -= 1;
        nearbyBucketIndices[7].y -= 1;
    }

    if ((originIndex.z + 0.5f) * _gridSpacing <= position.z) {
        nearbyBucketIndices[1].z += 1;
        nearbyBucketIndices[3].z += 1;
        nearbyBucketIndices[5].z += 1;
        nearbyBucketIndices[7].z += 1;
    } else {
        nearbyBucketIndices[1].z -= 1;
        nearbyBucketIndices[3].z -= 1;
        nearbyBucketIndices[5].z -= 1;
        nearbyBucketIndices[7].z -= 1;
    }

    for (int i = 0; i < 8; i++) {
        nearbyKeys[i] = getHashKeyFromBucketIndex(nearbyBucketIndices[i]);
    }
}

PointNeighborSearcher3Ptr PointParallelHashGridSearcher3::clone() const {
    return CLONE_W_CUSTOM_DELETER(PointParallelHashGridSearcher3);
}

PointParallelHashGridSearcher3&
PointParallelHashGridSearcher3::operator=(
    const PointParallelHashGridSearcher3& other) {
    set(other);
    return *this;
}

void PointParallelHashGridSearcher3::set(
    const PointParallelHashGridSearcher3& other) {
    _gridSpacing = other._gridSpacing;
    _resolution = other._resolution;
    _points = other._points;
    _keys = other._keys;
    _startIndexTable = other._startIndexTable;
    _endIndexTable = other._endIndexTable;
    _sortedIndices = other._sortedIndices;
}

void PointParallelHashGridSearcher3::serialize(
    std::vector<uint8_t>* buffer) const {
    flatbuffers::FlatBufferBuilder builder(1024);

    // Copy simple data
    auto fbsResolution
        = fbs::Size3(_resolution.x, _resolution.y, _resolution.z);

    // Copy points
    std::vector<fbs::Vector3D> points;
    for (const auto& pt : _points) {
        points.push_back(jetToFbs(pt));
    }

    auto fbsPoints
        = builder.CreateVectorOfStructs(points.data(), points.size());

    // Copy key/tables
    std::vector<uint64_t> keys(_keys.begin(), _keys.end());
    std::vector<uint64_t> startIndexTable(
        _startIndexTable.begin(), _startIndexTable.end());
    std::vector<uint64_t> endIndexTable(
        _endIndexTable.begin(), _endIndexTable.end());
    std::vector<uint64_t> sortedIndices(
        _sortedIndices.begin(), _sortedIndices.end());

    auto fbsKeys = builder.CreateVector(keys.data(), keys.size());
    auto fbsStartIndexTable
        = builder.CreateVector(startIndexTable.data(), startIndexTable.size());
    auto fbsEndIndexTable
        = builder.CreateVector(endIndexTable.data(), endIndexTable.size());
    auto fbsSortedIndices
        = builder.CreateVector(sortedIndices.data(), sortedIndices.size());

    // Copy the searcher
    auto fbsSearcher = fbs::CreatePointParallelHashGridSearcher3(
        builder,
        _gridSpacing,
        &fbsResolution,
        fbsPoints,
        fbsKeys,
        fbsStartIndexTable,
        fbsEndIndexTable,
        fbsSortedIndices);

    builder.Finish(fbsSearcher);

    uint8_t *buf = builder.GetBufferPointer();
    size_t size = builder.GetSize();

    buffer->resize(size);
    memcpy(buffer->data(), buf, size);
}

void PointParallelHashGridSearcher3::deserialize(
    const std::vector<uint8_t>& buffer) {
    auto fbsSearcher = fbs::GetPointParallelHashGridSearcher3(buffer.data());

    // Copy simple data
    auto res = fbsToJet(*fbsSearcher->resolution());
    _resolution.set({res.x, res.y, res.z});
    _gridSpacing = fbsSearcher->gridSpacing();

    // Copy points
    auto fbsPoints = fbsSearcher->points();
    _points.resize(fbsPoints->size());
    for (uint32_t i = 0; i < fbsPoints->size(); ++i) {
        _points[i] = fbsToJet(*fbsPoints->Get(i));
    }

    // Copy key/tables
    auto fbsKeys = fbsSearcher->keys();
    _keys.resize(fbsKeys->size());
    for (uint32_t i = 0; i < fbsKeys->size(); ++i) {
        _keys[i] = static_cast<size_t>(fbsKeys->Get(i));
    }

    auto fbsStartIndexTable = fbsSearcher->startIndexTable();
    _startIndexTable.resize(fbsStartIndexTable->size());
    for (uint32_t i = 0; i < fbsStartIndexTable->size(); ++i) {
        _startIndexTable[i] = static_cast<size_t>(fbsStartIndexTable->Get(i));
    }

    auto fbsEndIndexTable = fbsSearcher->endIndexTable();
    _endIndexTable.resize(fbsEndIndexTable->size());
    for (uint32_t i = 0; i < fbsEndIndexTable->size(); ++i) {
        _endIndexTable[i] = static_cast<size_t>(fbsEndIndexTable->Get(i));
    }

    auto fbsSortedIndices = fbsSearcher->sortedIndices();
    _sortedIndices.resize(fbsSortedIndices->size());
    for (uint32_t i = 0; i < fbsSortedIndices->size(); ++i) {
        _sortedIndices[i] = static_cast<size_t>(fbsSortedIndices->Get(i));
    }
}

PointParallelHashGridSearcher3::Builder
PointParallelHashGridSearcher3::builder() {
    return Builder();
}


PointParallelHashGridSearcher3::Builder&
PointParallelHashGridSearcher3::Builder::withResolution(
    const Size3& resolution) {
    _resolution = resolution;
    return *this;
}

PointParallelHashGridSearcher3::Builder&
PointParallelHashGridSearcher3::Builder::withGridSpacing(
    double gridSpacing) {
    _gridSpacing = gridSpacing;
    return *this;
}

PointParallelHashGridSearcher3
PointParallelHashGridSearcher3::Builder::build() const {
    return PointParallelHashGridSearcher3(_resolution, _gridSpacing);
}

PointParallelHashGridSearcher3Ptr
PointParallelHashGridSearcher3::Builder::makeShared() const {
    return std::shared_ptr<PointParallelHashGridSearcher3>(
        new PointParallelHashGridSearcher3(_resolution, _gridSpacing),
        [] (PointParallelHashGridSearcher3* obj) {
            delete obj;
        });
}

PointNeighborSearcher3Ptr
PointParallelHashGridSearcher3::Builder::buildPointNeighborSearcher() const {
    return makeShared();
}
