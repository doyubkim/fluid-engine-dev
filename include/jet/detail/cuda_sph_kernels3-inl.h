// Copyright (c) 2018 Doyub Kim
//
// I am making my contributions/submissions to this project solely in my
// personal capacity and am not conveying any rights to any intellectual
// property of any third parties.
//
// Adopted from the sample code of:
// Bart Adams and Martin Wicke,
// "Meshless Approximation Methods and Applications in Physics Based Modeling
// and Animation", Eurographics 2009 Tutorial

#ifdef JET_USE_CUDA

#ifndef INCLUDE_JET_DETAIL_CUDA_SPH_KERNELS3_INL_H_
#define INCLUDE_JET_DETAIL_CUDA_SPH_KERNELS3_INL_H_

#include <jet/constants.h>
#include <jet/cuda_sph_kernels3.h>
#include <jet/cuda_utils.h>

namespace jet {

namespace experimental {

inline CudaSphStdKernel3::CudaSphStdKernel3() : h(0), h2(0), h3(0), h5(0) {}

inline CudaSphStdKernel3::CudaSphStdKernel3(float kernelRadius)
    : h(kernelRadius), h2(h * h), h3(h2 * h), h5(h2 * h3) {}

inline CudaSphStdKernel3::CudaSphStdKernel3(const CudaSphStdKernel3& other)
    : h(other.h), h2(other.h2), h3(other.h3), h5(other.h5) {}

inline float CudaSphStdKernel3::operator()(float distance) const {
    if (distance * distance >= h2) {
        return 0.0f;
    } else {
        float x = 1.0f - distance * distance / h2;
        return 315.0f / (64.0f * kPiF * h3) * x * x * x;
    }
}

inline float CudaSphStdKernel3::firstDerivative(float distance) const {
    if (distance >= h) {
        return 0.0f;
    } else {
        float x = 1.0f - distance * distance / h2;
        return -945.0f / (32.0f * kPiF * h5) * distance * x * x;
    }
}

inline float4 CudaSphStdKernel3::gradient(const float4& point) const {
    float dist = length(point);
    if (dist > 0.0f) {
        return gradient(dist, point / dist);
    } else {
        return make_float4(0, 0, 0, 0);
    }
}

inline float4 CudaSphStdKernel3::gradient(
    float distance, const float4& directionToCenter) const {
    return -firstDerivative(distance) * directionToCenter;
}

inline float CudaSphStdKernel3::secondDerivative(float distance) const {
    if (distance * distance >= h2) {
        return 0.0f;
    } else {
        float x = distance * distance / h2;
        return 945.0f / (32.0f * kPiF * h5) * (1 - x) * (3 * x - 1);
    }
}

inline CudaSphSpikyKernel3::CudaSphSpikyKernel3() : h(0), h2(0), h3(0), h4(0), h5(0) {}

inline CudaSphSpikyKernel3::CudaSphSpikyKernel3(float h_)
    : h(h_), h2(h * h), h3(h2 * h), h4(h2 * h2), h5(h3 * h2) {}

inline CudaSphSpikyKernel3::CudaSphSpikyKernel3(const CudaSphSpikyKernel3& other)
    : h(other.h), h2(other.h2), h3(other.h3), h4(other.h4), h5(other.h5) {}

inline float CudaSphSpikyKernel3::operator()(float distance) const {
    if (distance >= h) {
        return 0.0f;
    } else {
        float x = 1.0f - distance / h;
        return 15.0f / (kPiF * h3) * x * x * x;
    }
}

inline float CudaSphSpikyKernel3::firstDerivative(float distance) const {
    if (distance >= h) {
        return 0.0f;
    } else {
        float x = 1.0f - distance / h;
        return -45.0f / (kPiF * h4) * x * x;
    }
}

inline float4 CudaSphSpikyKernel3::gradient(const float4& point) const {
    float dist = length(point);
    if (dist > 0.0f) {
        return gradient(dist, point / dist);
    } else {
        return make_float4(0, 0, 0, 0);
    }
}

inline float4 CudaSphSpikyKernel3::gradient(float distance,
                                        const float4& directionToCenter) const {
    return -firstDerivative(distance) * directionToCenter;
}

inline float CudaSphSpikyKernel3::secondDerivative(float distance) const {
    if (distance >= h) {
        return 0.0f;
    } else {
        float x = 1.0f - distance / h;
        return 90.0f / (kPiF * h5) * x;
    }
}

}  // namespace experimental

}  // namespace jet

#endif  // INCLUDE_JET_DETAIL_CUDA_SPH_KERNELS3_INL_H_

#endif  // JET_USE_CUDA
