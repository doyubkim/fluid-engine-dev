// Copyright (c) 2017 Doyub Kim
//
// I am making my contributions/submissions to this project solely in my
// personal capacity and am not conveying any rights to any intellectual
// property of any third parties.

#include <pch.h>

#ifdef JET_USE_CUDA

#include <jet/cuda_particle_system_solver3.h>

using namespace jet;
using namespace experimental;

CudaParticleSystemSolver3::CudaParticleSystemSolver3()
    : CudaParticleSystemSolver3(1e-3f, 1e-3f) {}

CudaParticleSystemSolver3::CudaParticleSystemSolver3(float radius, float mass)
    : _radius(radius), _mass(mass) {}

CudaParticleSystemSolver3::~CudaParticleSystemSolver3() {}

float CudaParticleSystemSolver3::radius() const { return _radius; }

void CudaParticleSystemSolver3::setRadius(float newRadius) {
    _radius = newRadius;
}

float CudaParticleSystemSolver3::mass() const { return _mass; }

void CudaParticleSystemSolver3::setMass(float newMass) { _mass = newMass; }

CudaParticleSystemSolver3::Builder CudaParticleSystemSolver3::builder() {
    return Builder();
}

//

CudaParticleSystemSolver3::Builder&
CudaParticleSystemSolver3::Builder::withRadius(float radius) {
    _radius = radius;
    return (*this);
}

CudaParticleSystemSolver3::Builder&
CudaParticleSystemSolver3::Builder::withMass(float mass) {
    _mass = mass;
    return (*this);
}

CudaParticleSystemSolver3 CudaParticleSystemSolver3::Builder::build() const {
    return CudaParticleSystemSolver3(_radius, _mass);
}

CudaParticleSystemSolver3Ptr CudaParticleSystemSolver3::Builder::makeShared()
    const {
    return std::shared_ptr<CudaParticleSystemSolver3>(
        new CudaParticleSystemSolver3(_radius, _mass),
        [](CudaParticleSystemSolver3* obj) { delete obj; });
}

#endif  // JET_USE_CUDA
