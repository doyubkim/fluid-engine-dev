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
#include <generated/point_parallel_hash_grid_searcher2_generated.h>
#include <generated/point_parallel_hash_grid_searcher3_generated.h>

#include <jet/constants.h>
#include <jet/parallel.h>
#include <jet/point_hash_grid_utils.h>
#include <jet/point_parallel_hash_grid_searcher.h>

namespace jet {

template <size_t N>
PointParallelHashGridSearcher<N>::PointParallelHashGridSearcher(
    const Vector<size_t, N> &resolution, double gridSpacing)
    : _gridSpacing(gridSpacing) {
    _resolution = max(resolution.template castTo<ssize_t>(),
                      Vector<ssize_t, N>::makeConstant(kOneSSize));
    size_t tableSize = static_cast<size_t>(product(_resolution, kOneSSize));
    _startIndexTable.resize(tableSize, kMaxSize);
    _endIndexTable.resize(tableSize, kMaxSize);
}

template <size_t N>
PointParallelHashGridSearcher<N>::PointParallelHashGridSearcher(
    const PointParallelHashGridSearcher &other) {
    set(other);
}

template <size_t N>
void PointParallelHashGridSearcher<N>::build(
    const ConstArrayView1<Vector<double, N>> &points) {
    _points.clear();
    _keys.clear();
    _startIndexTable.clear();
    _endIndexTable.clear();
    _sortedIndices.clear();

    // Allocate memory chunks
    size_t numberOfPoints = points.length();
    Array1<size_t> tempKeys(numberOfPoints);
    size_t tableSize = static_cast<size_t>(product(_resolution, kOneSSize));
    _startIndexTable.resize(tableSize);
    _endIndexTable.resize(tableSize);
    parallelFill(_startIndexTable.begin(), _startIndexTable.end(), kMaxSize);
    parallelFill(_endIndexTable.begin(), _endIndexTable.end(), kMaxSize);
    _keys.resize(numberOfPoints);
    _sortedIndices.resize(numberOfPoints);
    _points.resize(numberOfPoints);

    if (numberOfPoints == 0) {
        return;
    }

    // Initialize indices array and generate hash key for each point
    parallelFor(kZeroSize, numberOfPoints, [&](size_t i) {
        _sortedIndices[i] = i;
        _points[i] = points[i];
        tempKeys[i] = PointHashGridUtils<N>::getHashKeyFromPosition(
            points[i], _gridSpacing, _resolution);
    });

    // Sort indices based on hash key
    parallelSort(_sortedIndices.begin(), _sortedIndices.end(),
                 [&tempKeys](size_t indexA, size_t indexB) {
                     return tempKeys[indexA] < tempKeys[indexB];
                 });

    // Re-order point and key arrays
    parallelFor(kZeroSize, numberOfPoints, [&](size_t i) {
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

    parallelFor((size_t)1, numberOfPoints, [&](size_t i) {
        if (_keys[i] > _keys[i - 1]) {
            _startIndexTable[_keys[i]] = i;
            _endIndexTable[_keys[i - 1]] = i;
        }
    });

    size_t sumNumberOfPointsPerBucket = 0;
    size_t maxNumberOfPointsPerBucket = 0;
    size_t numberOfNonEmptyBucket = 0;
    for (size_t i = 0; i < _startIndexTable.length(); ++i) {
        if (_startIndexTable[i] != kMaxSize) {
            size_t numberOfPointsInBucket =
                _endIndexTable[i] - _startIndexTable[i];
            sumNumberOfPointsPerBucket += numberOfPointsInBucket;
            maxNumberOfPointsPerBucket =
                std::max(maxNumberOfPointsPerBucket, numberOfPointsInBucket);
            ++numberOfNonEmptyBucket;
        }
    }

    JET_INFO << "Average number of points per non-empty bucket: "
             << static_cast<float>(sumNumberOfPointsPerBucket) /
                    static_cast<float>(numberOfNonEmptyBucket);
    JET_INFO << "Max number of points per bucket: "
             << maxNumberOfPointsPerBucket;
}

template <size_t N>
void PointParallelHashGridSearcher<N>::build(
    const ConstArrayView1<Vector<double, N>> &points, double maxSearchRadius) {
    _gridSpacing = 2.0 * maxSearchRadius;

    build(points);
}

template <size_t N>
void PointParallelHashGridSearcher<N>::forEachNearbyPoint(
    const Vector<double, N> &origin, double radius,
    const ForEachNearbyPointFunc &callback) const {
    constexpr int kNumKeys = 1 << N;
    size_t nearbyKeys[kNumKeys];
    PointHashGridUtils<N>::getNearbyKeys(origin, _gridSpacing, _resolution,
                                         nearbyKeys);

    const double queryRadiusSquared = radius * radius;

    for (int i = 0; i < kNumKeys; i++) {
        size_t nearbyKey = nearbyKeys[i];
        size_t start = _startIndexTable[nearbyKey];
        size_t end = _endIndexTable[nearbyKey];

        // Empty bucket -- continue to next bucket
        if (start == kMaxSize) {
            continue;
        }

        for (size_t j = start; j < end; ++j) {
            Vector<double, N> direction = _points[j] - origin;
            double distanceSquared = direction.lengthSquared();
            if (distanceSquared <= queryRadiusSquared) {
                callback(_sortedIndices[j], _points[j]);
            }
        }
    }
}

template <size_t N>
bool PointParallelHashGridSearcher<N>::hasNearbyPoint(
    const Vector<double, N> &origin, double radius) const {
    constexpr int kNumKeys = 1 << N;
    size_t nearbyKeys[kNumKeys];
    PointHashGridUtils<N>::getNearbyKeys(origin, _gridSpacing, _resolution,
                                         nearbyKeys);

    const double queryRadiusSquared = radius * radius;

    for (int i = 0; i < kNumKeys; i++) {
        size_t nearbyKey = nearbyKeys[i];
        size_t start = _startIndexTable[nearbyKey];
        size_t end = _endIndexTable[nearbyKey];

        // Empty bucket -- continue to next bucket
        if (start == kMaxSize) {
            continue;
        }

        for (size_t j = start; j < end; ++j) {
            Vector<double, N> direction = _points[j] - origin;
            double distanceSquared = direction.lengthSquared();
            if (distanceSquared <= queryRadiusSquared) {
                return true;
            }
        }
    }

    return false;
}

template <size_t N>
ConstArrayView1<size_t> PointParallelHashGridSearcher<N>::keys() const {
    return _keys;
}

template <size_t N>
ConstArrayView1<size_t> PointParallelHashGridSearcher<N>::startIndexTable()
    const {
    return _startIndexTable;
}

template <size_t N>
ConstArrayView1<size_t> PointParallelHashGridSearcher<N>::endIndexTable()
    const {
    return _endIndexTable;
}

template <size_t N>
ConstArrayView1<size_t> PointParallelHashGridSearcher<N>::sortedIndices()
    const {
    return _sortedIndices;
}

template <size_t N>
std::shared_ptr<PointNeighborSearcher<N>>
PointParallelHashGridSearcher<N>::clone() const {
    return CLONE_W_CUSTOM_DELETER(PointParallelHashGridSearcher);
}

template <size_t N>
PointParallelHashGridSearcher<N> &PointParallelHashGridSearcher<N>::operator=(
    const PointParallelHashGridSearcher &other) {
    set(other);
    return *this;
}

template <size_t N>
void PointParallelHashGridSearcher<N>::set(
    const PointParallelHashGridSearcher &other) {
    _gridSpacing = other._gridSpacing;
    _resolution = other._resolution;
    _points = other._points;
    _keys = other._keys;
    _startIndexTable = other._startIndexTable;
    _endIndexTable = other._endIndexTable;
    _sortedIndices = other._sortedIndices;
}

template <size_t N>
void PointParallelHashGridSearcher<N>::serialize(
    std::vector<uint8_t> *buffer) const {
    serialize(*this, buffer);
}

template <size_t N>
void PointParallelHashGridSearcher<N>::deserialize(
    const std::vector<uint8_t> &buffer) {
    deserialize(buffer, *this);
}

template <size_t N>
typename PointParallelHashGridSearcher<N>::Builder
PointParallelHashGridSearcher<N>::builder() {
    return Builder();
}

template <size_t N>
template <size_t M>
std::enable_if_t<M == 2, void> PointParallelHashGridSearcher<N>::serialize(
    const PointParallelHashGridSearcher<2> &searcher,
    std::vector<uint8_t> *buffer) {
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

    // Copy key/tables
    std::vector<uint64_t> keys(searcher._keys.begin(), searcher._keys.end());
    std::vector<uint64_t> startIndexTable(searcher._startIndexTable.begin(),
                                          searcher._startIndexTable.end());
    std::vector<uint64_t> endIndexTable(searcher._endIndexTable.begin(),
                                        searcher._endIndexTable.end());
    std::vector<uint64_t> sortedIndices(searcher._sortedIndices.begin(),
                                        searcher._sortedIndices.end());

    auto fbsKeys = builder.CreateVector(keys.data(), keys.size());
    auto fbsStartIndexTable =
        builder.CreateVector(startIndexTable.data(), startIndexTable.size());
    auto fbsEndIndexTable =
        builder.CreateVector(endIndexTable.data(), endIndexTable.size());
    auto fbsSortedIndices =
        builder.CreateVector(sortedIndices.data(), sortedIndices.size());

    // Copy the searcher
    auto fbsSearcher = fbs::CreatePointParallelHashGridSearcher2(
        builder, searcher._gridSpacing, &fbsResolution, fbsPoints, fbsKeys,
        fbsStartIndexTable, fbsEndIndexTable, fbsSortedIndices);

    builder.Finish(fbsSearcher);

    uint8_t *buf = builder.GetBufferPointer();
    size_t size = builder.GetSize();

    buffer->resize(size);
    memcpy(buffer->data(), buf, size);
}

template <size_t N>
template <size_t M>
std::enable_if_t<M == 3, void> PointParallelHashGridSearcher<N>::serialize(
    const PointParallelHashGridSearcher<3> &searcher,
    std::vector<uint8_t> *buffer) {
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

    // Copy key/tables
    std::vector<uint64_t> keys(searcher._keys.begin(), searcher._keys.end());
    std::vector<uint64_t> startIndexTable(searcher._startIndexTable.begin(),
                                          searcher._startIndexTable.end());
    std::vector<uint64_t> endIndexTable(searcher._endIndexTable.begin(),
                                        searcher._endIndexTable.end());
    std::vector<uint64_t> sortedIndices(searcher._sortedIndices.begin(),
                                        searcher._sortedIndices.end());

    auto fbsKeys = builder.CreateVector(keys.data(), keys.size());
    auto fbsStartIndexTable =
        builder.CreateVector(startIndexTable.data(), startIndexTable.size());
    auto fbsEndIndexTable =
        builder.CreateVector(endIndexTable.data(), endIndexTable.size());
    auto fbsSortedIndices =
        builder.CreateVector(sortedIndices.data(), sortedIndices.size());

    // Copy the searcher
    auto fbsSearcher = fbs::CreatePointParallelHashGridSearcher3(
        builder, searcher._gridSpacing, &fbsResolution, fbsPoints, fbsKeys,
        fbsStartIndexTable, fbsEndIndexTable, fbsSortedIndices);

    builder.Finish(fbsSearcher);

    uint8_t *buf = builder.GetBufferPointer();
    size_t size = builder.GetSize();

    buffer->resize(size);
    memcpy(buffer->data(), buf, size);
}

template <size_t N>
template <size_t M>
std::enable_if_t<M == 2, void> PointParallelHashGridSearcher<N>::deserialize(
    const std::vector<uint8_t> &buffer,
    PointParallelHashGridSearcher<2> &searcher) {
    auto fbsSearcher = fbs::GetPointParallelHashGridSearcher2(buffer.data());

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

    // Copy key/tables
    auto fbsKeys = fbsSearcher->keys();
    searcher._keys.resize(fbsKeys->size());
    for (uint32_t i = 0; i < fbsKeys->size(); ++i) {
        searcher._keys[i] = static_cast<size_t>(fbsKeys->Get(i));
    }

    auto fbsStartIndexTable = fbsSearcher->startIndexTable();
    searcher._startIndexTable.resize(fbsStartIndexTable->size());
    for (uint32_t i = 0; i < fbsStartIndexTable->size(); ++i) {
        searcher._startIndexTable[i] =
            static_cast<size_t>(fbsStartIndexTable->Get(i));
    }

    auto fbsEndIndexTable = fbsSearcher->endIndexTable();
    searcher._endIndexTable.resize(fbsEndIndexTable->size());
    for (uint32_t i = 0; i < fbsEndIndexTable->size(); ++i) {
        searcher._endIndexTable[i] =
            static_cast<size_t>(fbsEndIndexTable->Get(i));
    }

    auto fbsSortedIndices = fbsSearcher->sortedIndices();
    searcher._sortedIndices.resize(fbsSortedIndices->size());
    for (uint32_t i = 0; i < fbsSortedIndices->size(); ++i) {
        searcher._sortedIndices[i] =
            static_cast<size_t>(fbsSortedIndices->Get(i));
    }
}

template <size_t N>
template <size_t M>
std::enable_if_t<M == 3, void> PointParallelHashGridSearcher<N>::deserialize(
    const std::vector<uint8_t> &buffer,
    PointParallelHashGridSearcher<3> &searcher) {
    auto fbsSearcher = fbs::GetPointParallelHashGridSearcher3(buffer.data());

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

    // Copy key/tables
    auto fbsKeys = fbsSearcher->keys();
    searcher._keys.resize(fbsKeys->size());
    for (uint32_t i = 0; i < fbsKeys->size(); ++i) {
        searcher._keys[i] = static_cast<size_t>(fbsKeys->Get(i));
    }

    auto fbsStartIndexTable = fbsSearcher->startIndexTable();
    searcher._startIndexTable.resize(fbsStartIndexTable->size());
    for (uint32_t i = 0; i < fbsStartIndexTable->size(); ++i) {
        searcher._startIndexTable[i] =
            static_cast<size_t>(fbsStartIndexTable->Get(i));
    }

    auto fbsEndIndexTable = fbsSearcher->endIndexTable();
    searcher._endIndexTable.resize(fbsEndIndexTable->size());
    for (uint32_t i = 0; i < fbsEndIndexTable->size(); ++i) {
        searcher._endIndexTable[i] =
            static_cast<size_t>(fbsEndIndexTable->Get(i));
    }

    auto fbsSortedIndices = fbsSearcher->sortedIndices();
    searcher._sortedIndices.resize(fbsSortedIndices->size());
    for (uint32_t i = 0; i < fbsSortedIndices->size(); ++i) {
        searcher._sortedIndices[i] =
            static_cast<size_t>(fbsSortedIndices->Get(i));
    }
}

template <size_t N>
typename PointParallelHashGridSearcher<N>::Builder &
PointParallelHashGridSearcher<N>::Builder::withResolution(
    const Vector<size_t, N> &resolution) {
    _resolution = resolution;
    return *this;
}

template <size_t N>
typename PointParallelHashGridSearcher<N>::Builder &
PointParallelHashGridSearcher<N>::Builder::withGridSpacing(double gridSpacing) {
    _gridSpacing = gridSpacing;
    return *this;
}

template <size_t N>
PointParallelHashGridSearcher<N>
PointParallelHashGridSearcher<N>::Builder::build() const {
    return PointParallelHashGridSearcher(_resolution, _gridSpacing);
}

template <size_t N>
std::shared_ptr<PointParallelHashGridSearcher<N>>
PointParallelHashGridSearcher<N>::Builder::makeShared() const {
    return std::shared_ptr<PointParallelHashGridSearcher>(
        new PointParallelHashGridSearcher(_resolution, _gridSpacing),
        [](PointParallelHashGridSearcher *obj) { delete obj; });
}

template <size_t N>
std::shared_ptr<PointNeighborSearcher<N>>
PointParallelHashGridSearcher<N>::Builder::buildPointNeighborSearcher() const {
    return makeShared();
}

template class PointParallelHashGridSearcher<2>;

template class PointParallelHashGridSearcher<3>;

}  // namespace jet
