// Copyright (c) 2017 Doyub Kim
//
// I am making my contributions/submissions to this project solely in my
// personal capacity and am not conveying any rights to any intellectual
// property of any third parties.

#include "cuda_particle_system_data3_func.h"

#include <jet/constants.h>

#include <cuda_runtime.h>

namespace jet {

namespace experimental {

inline JET_CUDA_HOST_DEVICE float stdKernel(float d2, float h2, float h3) {
    if (d2 >= h2) {
        return 0.0f;
    } else {
        float x = 1.0f - d2 / h2;
        return 315.0f / (64.0f * kPiF * h3) * x * x * x;
    }
}

struct UpdateDensity {
    float h2;
    float h3;
    float m;
    float* d;

    JET_CUDA_HOST_DEVICE UpdateDensity(float h, float m_, float* d_)
        : h2(h * h), h3(h * h * h), m(m_), d(d_) {}

    inline JET_CUDA_DEVICE void operator()(uint32_t i, float4 o, uint32_t,
                                           float4 p) {
        float dist = length(o - p);
        d[i] += m * stdKernel(dist, h2, h3);
    }
};

class BuildNeighborListsAndUpdateDensitiesFunc {
 public:
    inline JET_CUDA_HOST_DEVICE BuildNeighborListsAndUpdateDensitiesFunc(
        const uint32_t* neighborStarts, const uint32_t* neighborEnds, float h,
        float m, uint32_t* neighborLists, float* densities)
        : _neighborStarts(neighborStarts),
          _neighborEnds(neighborEnds),
          _h2(h * h),
          _h3(h * h * h),
          _mass(m),
          _neighborLists(neighborLists),
          _densities(densities) {}

    template <typename Index>
    inline JET_CUDA_HOST_DEVICE void operator()(size_t i, Index j, Index cnt,
                                                float d2) {
        if (cnt == 0) {
            _densities[i] = 0.0f;
        }

        _densities[i] += _mass * stdKernel(d2, _h2, _h3);

        if (i != j) {
            _neighborLists[_neighborStarts[i] + cnt] = j;
        }
    }

 private:
    const uint32_t* _neighborStarts;
    const uint32_t* _neighborEnds;
    float _h2;
    float _h3;
    float _mass;
    uint32_t* _neighborLists;
    float* _densities;
};

}  // namespace experimental

}  // namespace jet
