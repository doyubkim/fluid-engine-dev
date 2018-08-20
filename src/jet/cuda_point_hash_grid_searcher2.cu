// Copyright (c) 2018 Doyub Kim
//
// I am making my contributions/submissions to this project solely in my
// personal capacity and am not conveying any rights to any intellectual
// property of any third parties.

#include <jet/cuda_algorithms.h>
#include <jet/cuda_point_hash_grid_searcher2.h>

#include <thrust/for_each.h>
#include <thrust/sort.h>

using namespace jet;

namespace {

__global__ void initializeIndexTables(uint32_t* startIndexTable,
                                      uint32_t* endIndexTable, size_t n) {
    size_t i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) {
        startIndexTable[i] = 0xffffffff;
        endIndexTable[i] = 0xffffffff;
    }
}

__global__ void initializePointAndKeys(
    CudaPointHashGridSearcher2::HashUtils hashUtils, const float2* points,
    size_t n, uint32_t* sortedIndices, uint32_t* keys) {
    size_t i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) {
        sortedIndices[i] = i;
        keys[i] = hashUtils.getHashKeyFromPosition(points[i]);
    }
}

__global__ void buildTables(const uint32_t* keys, size_t n,
                            uint32_t* startIndexTable,
                            uint32_t* endIndexTable) {
    size_t i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n && i > 0) {
        uint32_t k = keys[i];
        uint32_t kLeft = keys[i - 1];
        if (k > kLeft) {
            startIndexTable[k] = i;
            endIndexTable[kLeft] = i;
        }
    }
}

}  // namespace

CudaPointHashGridSearcher2::CudaPointHashGridSearcher2(uint32_t resolutionX,
                                                       uint32_t resolutionY,
                                                       float gridSpacing)
    : _gridSpacing(gridSpacing) {
    _resolution.x = std::max(resolutionX, 1u);
    _resolution.y = std::max(resolutionY, 1u);

    _startIndexTable.resize(_resolution.x * _resolution.y, 0xffffffff);
    _endIndexTable.resize(_resolution.x * _resolution.y, 0xffffffff);
}

void CudaPointHashGridSearcher2::build(
    const ConstCudaArrayView1<float2>& points) {
    // Allocate/reset memory chuncks
    size_t numberOfPoints = points.length();
    if (numberOfPoints == 0) {
        return;
    }

    _points = points;

    // Initialize index tables
    size_t numberOfGrids = _startIndexTable.length();
    unsigned int numBlocks, numThreads;
    cudaComputeGridSize((unsigned int)numberOfGrids, 256, numBlocks,
                        numThreads);

    initializeIndexTables<<<numBlocks, numThreads>>>(_startIndexTable.data(),
                                                     _endIndexTable.data(),
                                                     _startIndexTable.length());

    // Initialize indices array and generate hash key for each point
    _keys.resize(numberOfPoints);
    _sortedIndices.resize(numberOfPoints);

    cudaComputeGridSize((unsigned int)numberOfPoints, 256, numBlocks,
                        numThreads);

    CudaPointHashGridSearcher2::HashUtils hashUtils(_gridSpacing, _resolution);

    initializePointAndKeys<<<numBlocks, numThreads>>>(
        hashUtils, _points.data(), _points.length(), _sortedIndices.data(),
        _keys.data());

    // Sort indices/points/key based on hash key
    thrust::device_ptr<uint32_t> keysBegin(_keys.data());
    thrust::device_ptr<uint32_t> keysEnd = keysBegin + _keys.length();
    thrust::device_ptr<float2> pointsBegin(_points.data());
    thrust::device_ptr<uint32_t> sortedIndicesBegin(_sortedIndices.data());
    thrust::sort_by_key(keysBegin, keysEnd,
                        thrust::make_zip_iterator(thrust::make_tuple(
                            pointsBegin, sortedIndicesBegin)));

    // Now _points and _keys are sorted by points' hash key values.
    // Let's fill in start/end index table with _keys.
    // Assume that _keys array looks like:
    // [5|8|8|10|10|10]
    // Then _startIndexTable and _endIndexTable should be like:
    // [.....|0|...|1|..|3|..]
    // [.....|1|...|3|..|6|..]
    //       ^5    ^8   ^10
    // So that _endIndexTable[i] - _startIndexTable[i] is the number points in
    // i-th table bucket.

    _startIndexTable[_keys[0]] = 0;
    _endIndexTable[_keys[numberOfPoints - 1]] = (uint32_t)numberOfPoints;

    cudaComputeGridSize((unsigned int)numberOfPoints, 256, numBlocks,
                        numThreads);

    buildTables<<<numBlocks, numThreads>>>(_keys.data(), numberOfPoints,
                                           _startIndexTable.data(),
                                           _endIndexTable.data());
}

void CudaPointHashGridSearcher2::set(const CudaPointHashGridSearcher2& other) {
    _gridSpacing = other._gridSpacing;
    _resolution = other._resolution;
    _points = other._points;
    _keys = other._keys;
    _startIndexTable = other._startIndexTable;
    _endIndexTable = other._endIndexTable;
    _sortedIndices = other._sortedIndices;
}
