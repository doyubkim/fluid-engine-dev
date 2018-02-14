// Copyright (c) 2017 Doyub Kim
//
// I am making my contributions/submissions to this project solely in my
// personal capacity and am not conveying any rights to any intellectual
// property of any third parties.

#include <pch.h>

#ifdef JET_USE_CUDA

#include <jet/cuda_wc_sph_solver3.h>

using namespace jet;
using namespace experimental;

CudaWcSphSolver3::CudaWcSphSolver3()
    : CudaWcSphSolver3(kWaterDensityF, 0.1f, 1.8f) {}

CudaWcSphSolver3::CudaWcSphSolver3(float targetDensity, float targetSpacing,
                                   float relativeKernelRadius)
    : CudaSphSolverBase3() {
    auto sph = sphSystemData();
    sph->setTargetDensity(targetDensity);
    sph->setTargetSpacing(targetSpacing);
    sph->setRelativeKernelRadius(relativeKernelRadius);
}

CudaWcSphSolver3::~CudaWcSphSolver3() {}

float CudaWcSphSolver3::eosExponent() const { return _eosExponent; }

void CudaWcSphSolver3::setEosExponent(float newEosExponent) {
    _eosExponent = std::max(newEosExponent, 1.0f);
}

CudaWcSphSolver3::Builder CudaWcSphSolver3::builder() { return Builder(); }

//

CudaWcSphSolver3 CudaWcSphSolver3::Builder::build() const {
    return CudaWcSphSolver3(_targetDensity, _targetSpacing,
                            _relativeKernelRadius);
}

CudaWcSphSolver3Ptr CudaWcSphSolver3::Builder::makeShared() const {
    return std::shared_ptr<CudaWcSphSolver3>(
        new CudaWcSphSolver3(_targetDensity, _targetSpacing,
                             _relativeKernelRadius),
        [](CudaWcSphSolver3* obj) { delete obj; });
}

#endif  // JET_USE_CUDA
