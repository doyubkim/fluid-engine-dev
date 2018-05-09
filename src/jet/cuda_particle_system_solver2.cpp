// Copyright (c) 2018 Doyub Kim
//
// I am making my contributions/submissions to this project solely in my
// personal capacity and am not conveying any rights to any intellectual
// property of any third parties.

#include <pch.h>

#ifdef JET_USE_CUDA

#include <jet/cuda_particle_system_solver2.h>

using namespace jet;

CudaParticleSystemSolver2::CudaParticleSystemSolver2()
    : CudaParticleSystemSolver2(1e-3f, 1e-3f) {}

CudaParticleSystemSolver2::CudaParticleSystemSolver2(float radius, float mass)
    : _radius(radius), _mass(mass) {}

CudaParticleSystemSolver2::~CudaParticleSystemSolver2() {}

float CudaParticleSystemSolver2::radius() const { return _radius; }

void CudaParticleSystemSolver2::setRadius(float newRadius) {
    _radius = newRadius;
}

float CudaParticleSystemSolver2::mass() const { return _mass; }

void CudaParticleSystemSolver2::setMass(float newMass) { _mass = newMass; }

CudaParticleSystemSolver2::Builder CudaParticleSystemSolver2::builder() {
    return Builder();
}

//

CudaParticleSystemSolver2::Builder&
CudaParticleSystemSolver2::Builder::withRadius(float radius) {
    _radius = radius;
    return (*this);
}

CudaParticleSystemSolver2::Builder&
CudaParticleSystemSolver2::Builder::withMass(float mass) {
    _mass = mass;
    return (*this);
}

CudaParticleSystemSolver2 CudaParticleSystemSolver2::Builder::build() const {
    return CudaParticleSystemSolver2(_radius, _mass);
}

CudaParticleSystemSolver2Ptr CudaParticleSystemSolver2::Builder::makeShared()
    const {
    return std::shared_ptr<CudaParticleSystemSolver2>(
        new CudaParticleSystemSolver2(_radius, _mass),
        [](CudaParticleSystemSolver2* obj) { delete obj; });
}

#endif  // JET_USE_CUDA
