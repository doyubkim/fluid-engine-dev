// Copyright (c) 2018 Doyub Kim
//
// I am making my contributions/submissions to this project solely in my
// personal capacity and am not conveying any rights to any intellectual
// property of any third parties.

#include <pch.h>

#ifdef JET_USE_CUDA

#include <jet/cuda_wc_sph_solver2.h>

using namespace jet;

CudaWcSphSolver2::CudaWcSphSolver2()
    : CudaWcSphSolver2(kWaterDensityF, 0.1f, 1.8f) {}

CudaWcSphSolver2::CudaWcSphSolver2(float targetDensity, float targetSpacing,
                                   float relativeKernelRadius)
    : CudaSphSolverBase2() {
    auto sph = sphSystemData();
    sph->setTargetDensity(targetDensity);
    sph->setTargetSpacing(targetSpacing);
    sph->setRelativeKernelRadius(relativeKernelRadius);
}

CudaWcSphSolver2::~CudaWcSphSolver2() {}

float CudaWcSphSolver2::eosExponent() const { return _eosExponent; }

void CudaWcSphSolver2::setEosExponent(float newEosExponent) {
    _eosExponent = std::max(newEosExponent, 1.0f);
}

CudaWcSphSolver2::Builder CudaWcSphSolver2::builder() { return Builder(); }

//

CudaWcSphSolver2 CudaWcSphSolver2::Builder::build() const {
    return CudaWcSphSolver2(_targetDensity, _targetSpacing,
                            _relativeKernelRadius);
}

CudaWcSphSolver2Ptr CudaWcSphSolver2::Builder::makeShared() const {
    return std::shared_ptr<CudaWcSphSolver2>(
        new CudaWcSphSolver2(_targetDensity, _targetSpacing,
                             _relativeKernelRadius),
        [](CudaWcSphSolver2* obj) { delete obj; });
}

#endif  // JET_USE_CUDA
