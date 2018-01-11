// Copyright (c) 2017 Doyub Kim
//
// I am making my contributions/submissions to this project solely in my
// personal capacity and am not conveying any rights to any intellectual
// property of any third parties.

#include <jet/cuda_point_hash_grid_searcher3.h>

#include <thrust/for_each.h>

using namespace jet;
using namespace experimental;

namespace {

inline __device__ int3 getBucketIndex(const float4& position,
                                      float gridSpacing) {
    int3 bucketIndex;
    bucketIndex.x = (int)(floor(position.x / gridSpacing));
    bucketIndex.y = (int)(floor(position.y / gridSpacing));
    bucketIndex.z = (int)(floor(position.z / gridSpacing));
    return bucketIndex;
}

inline __device__ size_t getHashKeyFromBucketIndex(const int3& bucketIndex,
                                                   const uint3& resolution) {
    int3 wrappedIndex = bucketIndex;
    wrappedIndex.x = bucketIndex.x % resolution.x;
    wrappedIndex.y = bucketIndex.y % resolution.y;
    wrappedIndex.z = bucketIndex.z % resolution.z;
    if (wrappedIndex.x < 0) {
        wrappedIndex.x += resolution.x;
    }
    if (wrappedIndex.y < 0) {
        wrappedIndex.y += resolution.y;
    }
    if (wrappedIndex.z < 0) {
        wrappedIndex.z += resolution.z;
    }
    return (size_t)((wrappedIndex.z * resolution.y + wrappedIndex.y) *
                        resolution.x +
                    wrappedIndex.x);
}

inline __device__ size_t getHashKeyFromPosition(const float4& position,
                                                const uint3& resolution,
                                                float gridSpacing) {
    int3 bucketIndex = getBucketIndex(position, gridSpacing);
    return getHashKeyFromBucketIndex(bucketIndex, resolution);
}

struct InitializeTables {
    template <typename Tuple>
    __device__ void operator()(Tuple t) {
        thrust::get<0>(t) = kMaxSize;
        thrust::get<1>(t) = kMaxSize;
    }
};

struct InitializeIndexPointAndKeys {
    float gridSpacing;
    uint3 resolution;

    template <typename Tuple>
    __device__ void operator()(Tuple t) {
        // 0: i [in]
        // 1: sortedIndices[out]
        // 2: points[in]
        // 3: points[out]
        // 4: keys[out]
        size_t i = thrust::get<0>(t);
        thrust::get<1>(t) = i;
        float4 p = thrust::get<2>(t);
        thrust::get<3>(t) = p;
        size_t key = getHashKeyFromPosition(p, resolution, gridSpacing);
        thrust::get<4>(t) = key;
    }
};

struct BuildTables {
    size_t* keys;
    size_t* startIndexTable;
    size_t* endIndexTable;

    template <typename Index>
    __device__ void operator()(Index i) {
        size_t k = keys[i];
        size_t kLeft = keys[i - 1];
        if (k > kLeft) {
            startIndexTable[k] = i;
            endIndexTable[kLeft] = i;
        }
    }
};

}  // namespace

CudaPointHashGridSearcher3::CudaPointHashGridSearcher3(const Size3& resolution,
                                                       float gridSpacing)
    : CudaPointHashGridSearcher3(resolution.x, resolution.y, resolution.z,
                                 gridSpacing) {}

CudaPointHashGridSearcher3::CudaPointHashGridSearcher3(size_t resolutionX,
                                                       size_t resolutionY,
                                                       size_t resolutionZ,
                                                       float gridSpacing)
    : _gridSpacing(gridSpacing) {
    _resolution.x = std::max(static_cast<ssize_t>(resolutionX), kOneSSize);
    _resolution.y = std::max(static_cast<ssize_t>(resolutionY), kOneSSize);
    _resolution.z = std::max(static_cast<ssize_t>(resolutionZ), kOneSSize);

    _startIndexTable.resize(_resolution.x * _resolution.y * _resolution.z,
                            kMaxSize);
    _endIndexTable.resize(_resolution.x * _resolution.y * _resolution.z,
                          kMaxSize);
}

CudaPointHashGridSearcher3::CudaPointHashGridSearcher3(
    const CudaPointHashGridSearcher3& other) {
    set(other);
}

void CudaPointHashGridSearcher3::build(const CudaArrayView1<float4>& points) {
    _points.clear();
    _keys.clear();
    _startIndexTable.clear();
    _endIndexTable.clear();
    _sortedIndices.clear();

    // Allocate memory chuncks
    size_t numberOfPoints = points.size();
    _startIndexTable.resize(_resolution.x * _resolution.y * _resolution.z);
    _endIndexTable.resize(_resolution.x * _resolution.y * _resolution.z);
    _keys.resize(numberOfPoints);
    _sortedIndices.resize(numberOfPoints);
    _points.resize(numberOfPoints);

    if (numberOfPoints == 0) {
        return;
    }

    // Initialize tables
    thrust::for_each(thrust::make_zip_iterator(thrust::make_tuple(
                         _startIndexTable.begin(), _endIndexTable.begin())),
                     thrust::make_zip_iterator(thrust::make_tuple(
                         _startIndexTable.end(), _endIndexTable.end())),
                     InitializeTables());

    // Initialize indices array and generate hash key for each point
    auto countingBegin = thrust::counting_iterator<size_t>(0);
    auto countingEnd = countingBegin + numberOfPoints;
    InitializeIndexPointAndKeys initIndexPointAndKeysFunc;
    initIndexPointAndKeysFunc.gridSpacing = _gridSpacing;
    initIndexPointAndKeysFunc.resolution =
        make_uint3((unsigned int)_resolution.x, (unsigned int)_resolution.y,
                   (unsigned int)_resolution.z);
    thrust::for_each(thrust::make_zip_iterator(thrust::make_tuple(
                         countingBegin, _sortedIndices.begin(), points.begin(),
                         _points.begin(), _keys.begin())),
                     thrust::make_zip_iterator(thrust::make_tuple(
                         countingEnd, _sortedIndices.end(), points.end(),
                         _points.end(), _keys.end())),
                     initIndexPointAndKeysFunc);

    // Sort indices/points/key based on hash key
    thrust::sort_by_key(_keys.begin(), _keys.end(),
                        thrust::make_zip_iterator(thrust::make_tuple(
                            _sortedIndices.begin(), _points.begin())));

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

    BuildTables buildTablesFunc;
    buildTablesFunc.keys = _keys.data();
    buildTablesFunc.startIndexTable = _startIndexTable.data();
    buildTablesFunc.endIndexTable = _endIndexTable.data();

    thrust::for_each(countingBegin + 1, countingEnd, buildTablesFunc);

#if 0
    size_t sumNumberOfPointsPerBucket = 0;
    size_t maxNumberOfPointsPerBucket = 0;
    size_t numberOfNonEmptyBucket = 0;
    for (size_t i = 0; i < _startIndexTable.size(); ++i) {
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
#endif
}

CudaArrayView1<size_t> CudaPointHashGridSearcher3::keys() const {
    return _keys.view();
}

CudaArrayView1<size_t> CudaPointHashGridSearcher3::startIndexTable() const {
    return _startIndexTable.view();
}

CudaArrayView1<size_t> CudaPointHashGridSearcher3::endIndexTable() const {
    return _endIndexTable.view();
}

CudaArrayView1<size_t> CudaPointHashGridSearcher3::sortedIndices() const {
    return _sortedIndices.view();
}

void CudaPointHashGridSearcher3::set(const CudaPointHashGridSearcher3& other) {
    _gridSpacing = other._gridSpacing;
    _resolution = other._resolution;
    _points = other._points;
    _keys = other._keys;
    _startIndexTable = other._startIndexTable;
    _endIndexTable = other._endIndexTable;
    _sortedIndices = other._sortedIndices;
}
