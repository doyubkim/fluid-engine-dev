// Copyright (c) 2017 Doyub Kim
//
// I am making my contributions/submissions to this project solely in my
// personal capacity and am not conveying any rights to any intellectual
// property of any third parties.

#include <jet/cuda_particle_system_data3.h>
#include <jet/macros.h>

#include <thrust/copy.h>

using namespace jet;
using namespace experimental;

namespace {

float4 toFloat4(const Vector4F& vec) {
    return make_float4(vec.x, vec.y, vec.z, vec.w);
}

}  // namespace

CudaParticleSystemData3::CudaParticleSystemData3()
    : CudaParticleSystemData3(0) {}

CudaParticleSystemData3::CudaParticleSystemData3(size_t numberOfParticles) {
    _positionIdx = addVectorData();
    _velocityIdx = addVectorData();

    resize(numberOfParticles);
}

CudaParticleSystemData3::CudaParticleSystemData3(
    const CudaParticleSystemData3& other) {
    set(other);
}

CudaParticleSystemData3::~CudaParticleSystemData3() {}

void CudaParticleSystemData3::resize(size_t newNumberOfParticles) {
    _numberOfParticles = newNumberOfParticles;

    for (auto& attr : _intDataList) {
        attr.resize(newNumberOfParticles, 0);
    }

    for (auto& attr : _floatDataList) {
        attr.resize(newNumberOfParticles, 0.0);
    }

    for (auto& attr : _vectorDataList) {
        attr.resize(newNumberOfParticles, make_float4(0, 0, 0, 0));
    }
}

size_t CudaParticleSystemData3::numberOfParticles() const {
    return _numberOfParticles;
}

size_t CudaParticleSystemData3::addIntData(int initialVal) {
    size_t attrIdx = _intDataList.size();
    _intDataList.emplace_back(numberOfParticles(), initialVal);
    return attrIdx;
}

size_t CudaParticleSystemData3::addFloatData(float initialVal) {
    size_t attrIdx = _floatDataList.size();
    _floatDataList.emplace_back(numberOfParticles(), initialVal);
    return attrIdx;
}

size_t CudaParticleSystemData3::addVectorData(const Vector4F& initialVal) {
    size_t attrIdx = _vectorDataList.size();
    _vectorDataList.emplace_back(numberOfParticles(), toFloat4(initialVal));
    return attrIdx;
}

size_t CudaParticleSystemData3::numberOfIntData() const {
    return _intDataList.size();
}

size_t CudaParticleSystemData3::numberOfFloatData() const {
    return _floatDataList.size();
}

size_t CudaParticleSystemData3::numberOfVectorData() const {
    return _vectorDataList.size();
}

CudaArrayView1<float4> CudaParticleSystemData3::positions() {
    return vectorDataAt(_positionIdx);
}

const CudaArrayView1<float4> CudaParticleSystemData3::positions() const {
    return vectorDataAt(_positionIdx);
}

CudaArrayView1<float4> CudaParticleSystemData3::velocities() {
    return vectorDataAt(_velocityIdx);
}

const CudaArrayView1<float4> CudaParticleSystemData3::velocities() const {
    return vectorDataAt(_velocityIdx);
}

CudaArrayView1<int> CudaParticleSystemData3::intDataAt(size_t idx) {
    return _intDataList[idx].view();
}

const CudaArrayView1<int> CudaParticleSystemData3::intDataAt(size_t idx) const {
    return _intDataList[idx].view();
}

CudaArrayView1<float> CudaParticleSystemData3::floatDataAt(size_t idx) {
    return _floatDataList[idx].view();
}

const CudaArrayView1<float> CudaParticleSystemData3::floatDataAt(
    size_t idx) const {
    return _floatDataList[idx].view();
}

CudaArrayView1<float4> CudaParticleSystemData3::vectorDataAt(size_t idx) {
    return _vectorDataList[idx].view();
}

const CudaArrayView1<float4> CudaParticleSystemData3::vectorDataAt(
    size_t idx) const {
    return _vectorDataList[idx].view();
}

void CudaParticleSystemData3::addParticle(const Vector4F& newPosition,
                                          const Vector4F& newVelocity) {
    thrust::host_vector<float4> hostPos;
    thrust::host_vector<float4> hostVel;
    hostPos.push_back(toFloat4(newPosition));
    hostVel.push_back(toFloat4(newVelocity));
    CudaArray1<float4> devicePos{hostPos};
    CudaArray1<float4> deviceVel{hostVel};

    addParticles(devicePos, deviceVel);
}

void CudaParticleSystemData3::addParticles(
    const ArrayView1<Vector4F>& newPositions,
    const ArrayView1<Vector4F>& newVelocities) {
    thrust::host_vector<float4> hostPos(newPositions.size());
    thrust::host_vector<float4> hostVel(newVelocities.size());
    for (size_t i = 0; i < newPositions.size(); ++i) {
        hostPos[i] = toFloat4(newPositions[i]);
    }
    for (size_t i = 0; i < newVelocities.size(); ++i) {
        hostVel[i] = toFloat4(newVelocities[i]);
    }

    CudaArray1<float4> devicePos{hostPos};
    CudaArray1<float4> deviceVel{hostVel};

    addParticles(devicePos, deviceVel);
}

void CudaParticleSystemData3::addParticles(
    const CudaArrayView1<float4>& newPositions,
    const CudaArrayView1<float4>& newVelocities) {
    JET_THROW_INVALID_ARG_IF(newVelocities.size() > 0 &&
                             newVelocities.size() != newPositions.size());

    size_t oldNumberOfParticles = numberOfParticles();
    size_t newNumberOfParticles = oldNumberOfParticles + newPositions.size();

    resize(newNumberOfParticles);

    auto pos = positions();

    thrust::copy(newPositions.begin(), newPositions.end(),
                 pos.begin() + oldNumberOfParticles);

    if (newVelocities.size() > 0) {
        auto vel = velocities();
        thrust::copy(newVelocities.begin(), newVelocities.end(),
                     vel.begin() + oldNumberOfParticles);
    }
}

void CudaParticleSystemData3::set(const CudaParticleSystemData3& other) {
    _numberOfParticles = other._numberOfParticles;
    _positionIdx = other._positionIdx;
    _velocityIdx = other._velocityIdx;

    _intDataList = other._intDataList;
    _floatDataList = other._floatDataList;
    _vectorDataList = other._vectorDataList;
}

CudaParticleSystemData3& CudaParticleSystemData3::operator=(
    const CudaParticleSystemData3& other) {
    set(other);
    return *this;
}
