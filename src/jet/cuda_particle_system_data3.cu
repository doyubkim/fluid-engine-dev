// Copyright (c) 2018 Doyub Kim
//
// I am making my contributions/submissions to this project solely in my
// personal capacity and am not conveying any rights to any intellectual
// property of any third parties.

#include "cuda_particle_system_data3_func.h"

#include <jet/cuda_algorithms.h>
#include <jet/cuda_particle_system_data3.h>
#include <jet/cuda_utils.h>
#include <jet/thrust_utils.h>
#include <jet/macros.h>

#include <thrust/copy.h>
#include <thrust/scan.h>
#include <thrust/transform.h>

using namespace jet;

constexpr uint32_t kDefaultHashGridResolution = 64;

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
        attr.resize(newNumberOfParticles, 0.0f);
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

ConstCudaArrayView1<float4> CudaParticleSystemData3::positions() const {
    return vectorDataAt(_positionIdx);
}

CudaArrayView1<float4> CudaParticleSystemData3::velocities() {
    return vectorDataAt(_velocityIdx);
}

ConstCudaArrayView1<float4> CudaParticleSystemData3::velocities() const {
    return vectorDataAt(_velocityIdx);
}

CudaArrayView1<int> CudaParticleSystemData3::intDataAt(size_t idx) {
    return _intDataList[idx].view();
}

ConstCudaArrayView1<int> CudaParticleSystemData3::intDataAt(size_t idx) const {
    return _intDataList[idx].view();
}

CudaArrayView1<float> CudaParticleSystemData3::floatDataAt(size_t idx) {
    return _floatDataList[idx].view();
}

ConstCudaArrayView1<float> CudaParticleSystemData3::floatDataAt(
    size_t idx) const {
    return _floatDataList[idx].view();
}

CudaArrayView1<float4> CudaParticleSystemData3::vectorDataAt(size_t idx) {
    return _vectorDataList[idx].view();
}

ConstCudaArrayView1<float4> CudaParticleSystemData3::vectorDataAt(
    size_t idx) const {
    return _vectorDataList[idx].view();
}

void CudaParticleSystemData3::addParticle(const Vector4F& newPosition,
                                          const Vector4F& newVelocity) {
    std::vector<float4> hostPos;
    std::vector<float4> hostVel;
    hostPos.push_back(toFloat4(newPosition));
    hostVel.push_back(toFloat4(newVelocity));
    CudaArray1<float4> devicePos{hostPos};
    CudaArray1<float4> deviceVel{hostVel};

    addParticles(devicePos, deviceVel);
}

void CudaParticleSystemData3::addParticles(
    ConstArrayView1<Vector4F> newPositions,
    ConstArrayView1<Vector4F> newVelocities) {
    std::vector<float4> hostPos(newPositions.length());
    std::vector<float4> hostVel(newVelocities.length());
    for (size_t i = 0; i < newPositions.length(); ++i) {
        hostPos[i] = toFloat4(newPositions[i]);
    }
    for (size_t i = 0; i < newVelocities.length(); ++i) {
        hostVel[i] = toFloat4(newVelocities[i]);
    }

    CudaArray1<float4> devicePos{hostPos};
    CudaArray1<float4> deviceVel{hostVel};

    addParticles(devicePos, deviceVel);
}

void CudaParticleSystemData3::addParticles(
    ConstCudaArrayView1<float4> newPositions,
    ConstCudaArrayView1<float4> newVelocities) {
    JET_THROW_INVALID_ARG_IF(newVelocities.length() > 0 &&
                             newVelocities.length() != newPositions.length());

    size_t oldNumberOfParticles = numberOfParticles();

    resize(oldNumberOfParticles + newPositions.length());

    auto pos = positions();

    cudaCopy(newPositions.data(), newPositions.length(),
             pos.data() + oldNumberOfParticles);

    if (newVelocities.length() > 0) {
        auto vel = velocities();

        cudaCopy(newVelocities.data(), newVelocities.length(),
                 vel.data() + oldNumberOfParticles);
    }
}

ConstCudaArrayView1<uint32_t> CudaParticleSystemData3::neighborStarts() const {
    return _neighborStarts.view();
}

ConstCudaArrayView1<uint32_t> CudaParticleSystemData3::neighborEnds() const {
    return _neighborEnds.view();
}

ConstCudaArrayView1<uint32_t> CudaParticleSystemData3::neighborLists() const {
    return _neighborLists.view();
}

const CudaPointHashGridSearcher3* CudaParticleSystemData3::neighborSearcher()
    const {
    return _neighborSearcher.get();
}

void CudaParticleSystemData3::buildNeighborSearcher(float maxSearchRadius) {
    if (_neighborSearcher == nullptr) {
        _neighborSearcher = std::make_shared<CudaPointHashGridSearcher3>(
            kDefaultHashGridResolution, kDefaultHashGridResolution,
            kDefaultHashGridResolution, 2.0f * maxSearchRadius);
    }
    _neighborSearcher->build(positions());
}

void CudaParticleSystemData3::buildNeighborLists(float maxSearchRadius) {
    _neighborStarts.resize(_numberOfParticles);
    _neighborEnds.resize(_numberOfParticles);

    auto neighborStarts = _neighborStarts.view();

     // Count nearby points
    thrust::for_each(
        thrust::counting_iterator<size_t>(0),
        thrust::counting_iterator<size_t>(0) + numberOfParticles(),
        ForEachNeighborFunc<NoOpFunc, CountNearbyPointsFunc>(
            *_neighborSearcher, maxSearchRadius, positions().data(), NoOpFunc(),
            CountNearbyPointsFunc(_neighborStarts.data())));

    // Make start/end point of neighbor list, and allocate neighbor list.
    thrust::inclusive_scan(thrustCBegin(_neighborStarts),
                           thrustCEnd(_neighborStarts),
                           thrustBegin(_neighborEnds));
    thrust::transform(thrustCBegin(_neighborEnds), thrustCEnd(_neighborEnds),
                      thrustCBegin(_neighborStarts),
                      thrustBegin(_neighborStarts),
                      thrust::minus<unsigned int>());
    size_t rbeginIdx =
        _neighborEnds.length() > 0 ? _neighborEnds.length() - 1 : 0;
    uint32_t m = _neighborEnds[rbeginIdx];
    _neighborLists.resize(m, 0);

    // Build neighbor lists
    thrust::for_each(
        thrust::counting_iterator<size_t>(0),
        thrust::counting_iterator<size_t>(0) + numberOfParticles(),
        ForEachNeighborFunc<BuildNeighborListsFunc, NoOpFunc>(
            *_neighborSearcher, maxSearchRadius, positions().data(),
            BuildNeighborListsFunc(_neighborStarts.data(), _neighborEnds.data(),
                                   _neighborLists.data()),
            NoOpFunc()));
}

void CudaParticleSystemData3::set(const CudaParticleSystemData3& other) {
    _numberOfParticles = other._numberOfParticles;
    _positionIdx = other._positionIdx;
    _velocityIdx = other._velocityIdx;

    _intDataList = other._intDataList;
    _floatDataList = other._floatDataList;
    _vectorDataList = other._vectorDataList;

    if (other._neighborSearcher != nullptr) {
        _neighborSearcher = std::make_shared<CudaPointHashGridSearcher3>(
            *other._neighborSearcher);
    }
    _neighborStarts = other._neighborStarts;
    _neighborEnds = other._neighborEnds;
    _neighborLists = other._neighborLists;
}

CudaParticleSystemData3& CudaParticleSystemData3::operator=(
    const CudaParticleSystemData3& other) {
    set(other);
    return *this;
}
