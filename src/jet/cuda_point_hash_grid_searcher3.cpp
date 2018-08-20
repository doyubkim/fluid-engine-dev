// Copyright (c) 2018 Doyub Kim
//
// I am making my contributions/submissions to this project solely in my
// personal capacity and am not conveying any rights to any intellectual
// property of any third parties.

#include <jet/cuda_point_hash_grid_searcher3.h>

using namespace jet;

CudaPointHashGridSearcher3::CudaPointHashGridSearcher3(const uint3& resolution,
                                                       float gridSpacing)
    : CudaPointHashGridSearcher3(resolution.x, resolution.y, resolution.z,
                                 gridSpacing) {}

CudaPointHashGridSearcher3::CudaPointHashGridSearcher3(
    const CudaPointHashGridSearcher3& other) {
    set(other);
}

float CudaPointHashGridSearcher3::gridSpacing() const { return _gridSpacing; }

Vector3UZ CudaPointHashGridSearcher3::resolution() const {
    return Vector3UZ{static_cast<uint32_t>(_resolution.x),
                     static_cast<uint32_t>(_resolution.y),
                     static_cast<uint32_t>(_resolution.z)};
}

ConstCudaArrayView1<float4> CudaPointHashGridSearcher3::sortedPoints() const {
    return _points;
}

ConstCudaArrayView1<uint32_t> CudaPointHashGridSearcher3::keys() const {
    return _keys.view();
}

ConstCudaArrayView1<uint32_t> CudaPointHashGridSearcher3::startIndexTable()
    const {
    return _startIndexTable.view();
}

ConstCudaArrayView1<uint32_t> CudaPointHashGridSearcher3::endIndexTable()
    const {
    return _endIndexTable.view();
}

ConstCudaArrayView1<uint32_t> CudaPointHashGridSearcher3::sortedIndices()
    const {
    return _sortedIndices.view();
}

CudaPointHashGridSearcher3& CudaPointHashGridSearcher3::operator=(
    const CudaPointHashGridSearcher3& other) {
    set(other);
    return (*this);
}
