// Copyright (c) 2018 Doyub Kim
//
// I am making my contributions/submissions to this project solely in my
// personal capacity and am not conveying any rights to any intellectual
// property of any third parties.

#include "cuda_sph_system_data2_func.h"

#include <jet/cuda_algorithms.h>
#include <jet/cuda_sph_system_data2.h>
#include <jet/triangle_point_generator.h>

#include <thrust/extrema.h>
#include <thrust/for_each.h>

using namespace jet;

void CudaSphSystemData2::updateDensities() {
    neighborSearcher()->forEachNearbyPoint(
        positions(), _kernelRadius,
        UpdateDensity(_kernelRadius, _mass, densities().data()));
}

void CudaSphSystemData2::buildNeighborListsAndUpdateDensities() {
    size_t n = numberOfParticles();

    _neighborStarts.resize(n);
    _neighborEnds.resize(n);

    auto neighborStarts = _neighborStarts.view();

    // Count nearby points
    thrust::for_each(
        thrust::counting_iterator<size_t>(0),
        thrust::counting_iterator<size_t>(0) + numberOfParticles(),
        ForEachNeighborFunc<NoOpFunc, CountNearbyPointsFunc>(
            *_neighborSearcher, _kernelRadius, positions().data(), NoOpFunc(),
            CountNearbyPointsFunc(_neighborStarts.data())));

    // Make start/end point of neighbor list, and allocate neighbor list.
    thrust::device_ptr<uint32_t> neighborStartsBegin(_neighborStarts.data());
    thrust::device_ptr<uint32_t> neighborStartsEnd =
        neighborStartsBegin + _neighborStarts.length();
    thrust::device_ptr<uint32_t> neighborEndsBegin(_neighborEnds.data());
    thrust::device_ptr<uint32_t> neighborEndsEnd =
        neighborEndsBegin + _neighborEnds.length();

    thrust::inclusive_scan(neighborStartsBegin, neighborStartsEnd,
                           neighborEndsBegin);
    thrust::transform(neighborEndsBegin, neighborEndsEnd, neighborStartsBegin,
                      neighborStartsBegin, thrust::minus<uint32_t>());
    size_t rbeginIdx = _neighborEnds.length() > 0 ? _neighborEnds.length() - 1 : 0;
    uint32_t m = _neighborEnds[rbeginIdx];
    _neighborLists.resize(m, 0);

    // Build neighbor lists and update densities
    auto d = densities();
    cudaFill(d.data(), d.length(), 0.0f);
    thrust::for_each(
        thrust::counting_iterator<size_t>(0),
        thrust::counting_iterator<size_t>(0) + numberOfParticles(),
        ForEachNeighborFunc<BuildNeighborListsAndUpdateDensitiesFunc, NoOpFunc>(
            *_neighborSearcher, _kernelRadius, positions().data(),
            BuildNeighborListsAndUpdateDensitiesFunc(
                _neighborStarts.data(), _neighborEnds.data(), _kernelRadius,
                _mass, _neighborLists.data(), d.data()),
            NoOpFunc()));
}
