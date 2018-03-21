// Copyright (c) 2018 Doyub Kim
//
// I am making my contributions/submissions to this project solely in my
// personal capacity and am not conveying any rights to any intellectual
// property of any third parties.

#include "cuda_particle_system_data3_func.h"

#include <jet/cuda_sph_kernels3.h>

#include <cuda_runtime.h>

namespace jet {

namespace experimental {

class UpdateDensity {
 public:
    JET_CUDA_HOST_DEVICE UpdateDensity(float h, float m, float* d)
        : _m(m), _d(d), _stdKernel(h) {}

    inline JET_CUDA_DEVICE void operator()(uint32_t i, float4 o, uint32_t,
                                           float4 p) {
        float dist = length(o - p);
        _d[i] += _m * _stdKernel(dist);
    }

 private:
    float _m;
    float* _d;
    CudaSphStdKernel3 _stdKernel;
};

class BuildNeighborListsAndUpdateDensitiesFunc {
 public:
    inline JET_CUDA_HOST_DEVICE BuildNeighborListsAndUpdateDensitiesFunc(
        const uint32_t* neighborStarts, const uint32_t* neighborEnds, float h,
        float m, uint32_t* neighborLists, float* densities)
        : _neighborStarts(neighborStarts),
          _neighborEnds(neighborEnds),
          _mass(m),
          _neighborLists(neighborLists),
          _densities(densities),
          _stdKernel(h) {}

    template <typename Index>
    inline JET_CUDA_HOST_DEVICE void operator()(size_t i, Index j, Index cnt,
                                                float d2) {
        _densities[i] += _mass * _stdKernel(sqrt(d2));

        if (i != j) {
            _neighborLists[_neighborStarts[i] + cnt] = j;
        }
    }

 private:
    const uint32_t* _neighborStarts;
    const uint32_t* _neighborEnds;
    float _mass;
    uint32_t* _neighborLists;
    float* _densities;
    CudaSphStdKernel3 _stdKernel;
};

}  // namespace experimental

}  // namespace jet
