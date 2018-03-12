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

#ifndef INCLUDE_JET_DETAIL_CUDA_SPH_KERNELS2_INL_H_
#define INCLUDE_JET_DETAIL_CUDA_SPH_KERNELS2_INL_H_

#include <jet/constants.h>
#include <jet/cuda_sph_kernels2.h>
#include <jet/cuda_utils.h>

namespace jet {

namespace experimental {

inline CudaSphStdKernel2::CudaSphStdKernel2() : h(0), h2(0), h3(0), h4(0) {}

inline CudaSphStdKernel2::CudaSphStdKernel2(float h_)
    : h(h_), h2(h * h), h3(h2 * h), h4(h2 * h2) {}

inline CudaSphStdKernel2::CudaSphStdKernel2(const CudaSphStdKernel2& other)
    : h(other.h), h2(other.h2), h3(other.h3), h4(other.h4) {}

inline float CudaSphStdKernel2::operator()(float distance) const {
    float distanceSquared = distance * distance;

    if (distanceSquared >= h2) {
        return 0.0f;
    } else {
        float x = 1.0f - distanceSquared / h2;
        return 4.0f / (kPiF * h2) * x * x * x;
    }
}

inline float CudaSphStdKernel2::firstDerivative(float distance) const {
    if (distance >= h) {
        return 0.0f;
    } else {
        float x = 1.0f - distance * distance / h2;
        return -24.0f * distance / (kPiF * h4) * x * x;
    }
}

inline float2 CudaSphStdKernel2::gradient(const float2& point) const {
    float dist = length(point);
    if (dist > 0.0f) {
        return gradient(dist, point / dist);
    } else {
        return make_float2(0, 0);
    }
}

inline float2 CudaSphStdKernel2::gradient(
    float distance, const float2& directionToCenter) const {
    return -firstDerivative(distance) * directionToCenter;
}

inline float CudaSphStdKernel2::secondDerivative(float distance) const {
    float distanceSquared = distance * distance;

    if (distanceSquared >= h2) {
        return 0.0f;
    } else {
        float x = distanceSquared / h2;
        return 24.0f / (kPiF * h4) * (1 - x) * (5 * x - 1);
    }
}

inline CudaSphSpikyKernel2::CudaSphSpikyKernel2()
    : h(0), h2(0), h3(0), h4(0), h5(0) {}

inline CudaSphSpikyKernel2::CudaSphSpikyKernel2(float h_)
    : h(h_), h2(h * h), h3(h2 * h), h4(h2 * h2), h5(h3 * h2) {}

inline CudaSphSpikyKernel2::CudaSphSpikyKernel2(
    const CudaSphSpikyKernel2& other)
    : h(other.h), h2(other.h2), h3(other.h3), h4(other.h4), h5(other.h5) {}

inline float CudaSphSpikyKernel2::operator()(float distance) const {
    if (distance >= h) {
        return 0.0f;
    } else {
        float x = 1.0f - distance / h;
        return 10.0f / (kPiF * h2) * x * x * x;
    }
}

inline float CudaSphSpikyKernel2::firstDerivative(float distance) const {
    if (distance >= h) {
        return 0.0f;
    } else {
        float x = 1.0f - distance / h;
        return -30.0f / (kPiF * h3) * x * x;
    }
}

inline float2 CudaSphSpikyKernel2::gradient(const float2& point) const {
    float dist = length(point);
    if (dist > 0.0f) {
        return gradient(dist, point / dist);
    } else {
        return make_float2(0, 0);
    }
}

inline float2 CudaSphSpikyKernel2::gradient(
    float distance, const float2& directionToCenter) const {
    return -firstDerivative(distance) * directionToCenter;
}

inline float CudaSphSpikyKernel2::secondDerivative(float distance) const {
    if (distance >= h) {
        return 0.0f;
    } else {
        float x = 1.0f - distance / h;
        return 60.0f / (kPiF * h4) * x;
    }
}

}  // namespace experimental

}  // namespace jet

#endif  // INCLUDE_JET_DETAIL_CUDA_SPH_KERNELS2_INL_H_

#endif  // JET_USE_CUDA
