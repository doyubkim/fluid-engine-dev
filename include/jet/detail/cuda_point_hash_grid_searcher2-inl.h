// Copyright (c) 2018 Doyub Kim
//
// I am making my contributions/submissions to this project solely in my
// personal capacity and am not conveying any rights to any intellectual
// property of any third parties.

#ifndef INCLUDE_JET_DETAIL_CUDA_POINT_HASH_GRID_SEARCHER2_INL_H_
#define INCLUDE_JET_DETAIL_CUDA_POINT_HASH_GRID_SEARCHER2_INL_H_

#ifdef JET_USE_CUDA

#include <jet/cuda_point_hash_grid_searcher2.h>
#include <jet/cuda_utils.h>

namespace jet {

namespace experimental {

JET_CUDA_HOST_DEVICE CudaPointHashGridSearcher2::HashUtils::HashUtils() {}

JET_CUDA_HOST_DEVICE CudaPointHashGridSearcher2::HashUtils::HashUtils(
    float gridSpacing, uint2 resolution)
    : _gridSpacing(gridSpacing), _resolution(resolution) {}

inline JET_CUDA_HOST_DEVICE void
CudaPointHashGridSearcher2::HashUtils::getNearbyKeys(
    float2 position, uint32_t* nearbyKeys) const {
    int2 originIndex = getBucketIndex(position), nearbyBucketIndices[8];

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

inline JET_CUDA_HOST_DEVICE int2
CudaPointHashGridSearcher2::HashUtils::getBucketIndex(float2 position) const {
    int2 bucketIndex;
    bucketIndex.x = (int)floorf(position.x / _gridSpacing);
    bucketIndex.y = (int)floorf(position.y / _gridSpacing);
    return bucketIndex;
}

inline JET_CUDA_HOST_DEVICE uint32_t
CudaPointHashGridSearcher2::HashUtils::getHashKeyFromBucketIndex(
    int2 bucketIndex) const {
    // Assumes _resolution is power of two
    bucketIndex.x = bucketIndex.x & (_resolution.x - 1);
    bucketIndex.y = bucketIndex.y & (_resolution.y - 1);
    return bucketIndex.y * _resolution.x + bucketIndex.x;
}

inline JET_CUDA_HOST_DEVICE uint32_t
CudaPointHashGridSearcher2::HashUtils::getHashKeyFromPosition(
    float2 position) const {
    int2 bucketIndex = getBucketIndex(position);
    return getHashKeyFromBucketIndex(bucketIndex);
}

template <typename Callback>
inline JET_CUDA_HOST_DEVICE CudaPointHashGridSearcher2::ForEachNearbyPointFunc<
    Callback>::ForEachNearbyPointFunc(float r, float gridSpacing,
                                      uint2 resolution, const uint32_t* sit,
                                      const uint32_t* eit, const uint32_t* si,
                                      const float2* p, const float2* o,
                                      Callback cb)
    : _hashUtils(gridSpacing, resolution),
      _radius(r),
      _startIndexTable(sit),
      _endIndexTable(eit),
      _sortedIndices(si),
      _points(p),
      _origins(o),
      _callback(cb) {}

template <typename Callback>
template <typename Index>
inline JET_CUDA_HOST_DEVICE void
CudaPointHashGridSearcher2::ForEachNearbyPointFunc<Callback>::operator()(
    Index idx) {
    const float2 origin = _origins[idx];

    uint32_t nearbyKeys[4];
    _hashUtils.getNearbyKeys(origin, nearbyKeys);

    const float queryRadiusSquared = _radius * _radius;

    for (int i = 0; i < 4; i++) {
        uint32_t nearbyKey = nearbyKeys[i];
        uint32_t start = _startIndexTable[nearbyKey];

        // Empty bucket -- continue to next bucket
        if (start == 0xffffffff) {
            continue;
        }

        uint32_t end = _endIndexTable[nearbyKey];

        for (uint32_t jj = start; jj < end; ++jj) {
            uint32_t j = _sortedIndices[jj];
            float2 p = _points[jj];
            float2 direction = p - origin;
            float distanceSquared = lengthSquared(direction);
            if (distanceSquared <= queryRadiusSquared) {
                float distance = 0.0f;
                if (distanceSquared > 0) {
                    distance = sqrtf(distanceSquared);
                    direction /= distance;
                }

                _callback(idx, origin, j, p);
            }
        }
    }
}

//

template <typename Callback>
void CudaPointHashGridSearcher2::forEachNearbyPoint(
    const ConstCudaArrayView1<float2>& origins, float radius,
    Callback callback) const {
    thrust::for_each(
        thrust::counting_iterator<size_t>(0),
        thrust::counting_iterator<size_t>(0) + origins.size(),
        ForEachNearbyPointFunc<Callback>(
            radius, _gridSpacing, _resolution, _startIndexTable.data(),
            _endIndexTable.data(), _sortedIndices.data(), _points.data(),
            origins.data(), callback));
}

}  // namespace experimental

}  // namespace jet

#endif  // JET_USE_CUDA

#endif  // INCLUDE_JET_DETAIL_CUDA_POINT_HASH_GRID_SEARCHER2_INL_H_
