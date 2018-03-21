// Copyright (c) 2018 Doyub Kim
//
// I am making my contributions/submissions to this project solely in my
// personal capacity and am not conveying any rights to any intellectual
// property of any third parties.

#include <jet/cuda_point_hash_grid_searcher3.h>
#include <jet/cuda_utils.h>
#include <jet/macros.h>

namespace jet {

namespace experimental {

template <typename NeighborCallback, typename NeighborCounterCallback>
class ForEachNeighborFunc {
 public:
    inline JET_CUDA_HOST ForEachNeighborFunc(
        const CudaPointHashGridSearcher3& searcher, float radius,
        const float4* origins, NeighborCallback insideCb,
        NeighborCounterCallback cntCb)
        : _neighborCallback(insideCb),
          _neighborNeighborCounterCallback(cntCb),
          _hashUtils(searcher.gridSpacing(), toUInt3(searcher.resolution())),
          _radius(radius),
          _startIndexTable(searcher.startIndexTable().data()),
          _endIndexTable(searcher.endIndexTable().data()),
          _sortedIndices(searcher.sortedIndices().data()),
          _points(searcher.sortedPoints().data()),
          _origins(origins) {}

    template <typename Index>
    inline JET_CUDA_HOST_DEVICE void operator()(Index i) {
        const float4 origin = _origins[i];

        uint32_t nearbyKeys[8];
        _hashUtils.getNearbyKeys(origin, nearbyKeys);

        const float queryRadiusSquared = _radius * _radius;

        uint32_t cnt = 0;
        for (int c = 0; c < 8; c++) {
            uint32_t nearbyKey = nearbyKeys[c];
            uint32_t start = _startIndexTable[nearbyKey];

            // Empty bucket -- continue to next bucket
            if (start == 0xffffffff) {
                continue;
            }

            uint32_t end = _endIndexTable[nearbyKey];

            for (uint32_t jj = start; jj < end; ++jj) {
                uint32_t j = _sortedIndices[jj];
                float4 r = _points[jj] - origin;
                float distanceSquared = lengthSquared(r);
                if (distanceSquared <= queryRadiusSquared) {
                    _neighborCallback(i, j, cnt, distanceSquared);
                    if (i != j) {
                        ++cnt;
                    }
                }
            }
        }

        _neighborNeighborCounterCallback(i, cnt);
    }

 private:
    NeighborCallback _neighborCallback;
    NeighborCounterCallback _neighborNeighborCounterCallback;
    CudaPointHashGridSearcher3::HashUtils _hashUtils;
    float _radius;
    const uint32_t* _startIndexTable;
    const uint32_t* _endIndexTable;
    const uint32_t* _sortedIndices;
    const float4* _points;
    const float4* _origins;
};

class NoOpFunc {
 public:
    template <typename Index>
    inline JET_CUDA_HOST_DEVICE void operator()(size_t, Index) {}

    template <typename Index>
    inline JET_CUDA_HOST_DEVICE void operator()(size_t, Index, Index, float) {}
};

class BuildNeighborListsFunc {
 public:
    inline JET_CUDA_HOST_DEVICE BuildNeighborListsFunc(
        const uint32_t* neighborStarts, const uint32_t* neighborEnds,
        uint32_t* neighborLists)
        : _neighborStarts(neighborStarts),
          _neighborEnds(neighborEnds),
          _neighborLists(neighborLists) {}

    template <typename Index>
    inline JET_CUDA_HOST_DEVICE void operator()(size_t i, Index j, Index cnt,
                                                float) {
        if (i != j) {
            _neighborLists[_neighborStarts[i] + cnt] = j;
        }
    }

 private:
    const uint32_t* _neighborStarts;
    const uint32_t* _neighborEnds;
    uint32_t* _neighborLists;
};

class CountNearbyPointsFunc {
 public:
    inline JET_CUDA_HOST_DEVICE CountNearbyPointsFunc(uint32_t* cnt)
        : _counts(cnt) {}

    template <typename Index>
    inline JET_CUDA_HOST_DEVICE void operator()(size_t idx, Index cnt) {
        _counts[idx] = cnt;
    }

 private:
    uint32_t* _counts;
};

}  // namespace experimental

}  // namespace jet
