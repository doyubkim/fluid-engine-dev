// Copyright (c) 2018 Doyub Kim
//
// I am making my contributions/submissions to this project solely in my
// personal capacity and am not conveying any rights to any intellectual
// property of any third parties.

#if JET_USE_CUDA

#include <jet/cuda_point_hash_grid_searcher.h>

using namespace jet;

CudaPointHashGridSearcher2::CudaPointHashGridSearcher2(const uint2& resolution,
                                                       float gridSpacing)
    : CudaPointHashGridSearcher2(resolution.x, resolution.y, gridSpacing) {}

CudaPointHashGridSearcher2::CudaPointHashGridSearcher2(
    const CudaPointHashGridSearcher2& other) {
    set(other);
}

float CudaPointHashGridSearcher2::gridSpacing() const { return _gridSpacing; }

Vector2UZ CudaPointHashGridSearcher2::resolution() const {
    return Vector2UZ{static_cast<uint32_t>(_resolution.x),
                     static_cast<uint32_t>(_resolution.y)};
}

ConstCudaArrayView1<float2> CudaPointHashGridSearcher2::sortedPoints() const {
    return _points;
}

ConstCudaArrayView1<uint32_t> CudaPointHashGridSearcher2::keys() const {
    return _keys.view();
}

ConstCudaArrayView1<uint32_t> CudaPointHashGridSearcher2::startIndexTable()
    const {
    return _startIndexTable.view();
}

ConstCudaArrayView1<uint32_t> CudaPointHashGridSearcher2::endIndexTable()
    const {
    return _endIndexTable.view();
}

ConstCudaArrayView1<uint32_t> CudaPointHashGridSearcher2::sortedIndices()
    const {
    return _sortedIndices.view();
}

CudaPointHashGridSearcher2& CudaPointHashGridSearcher2::operator=(
    const CudaPointHashGridSearcher2& other) {
    set(other);
    return (*this);
}

#endif  // JET_USE_CUDA
