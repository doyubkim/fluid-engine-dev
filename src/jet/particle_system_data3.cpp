// Copyright (c) 2016 Doyub Kim

#include <pch.h>
#include <jet/parallel.h>
#include <jet/particle_system_data3.h>
#include <jet/point_parallel_hash_grid_searcher3.h>
#include <jet/timer.h>

#include <algorithm>
#include <vector>

using namespace jet;

static const size_t kDefaultHashGridResolution = 64;

ParticleSystemData3::ParticleSystemData3() {
}

ParticleSystemData3::~ParticleSystemData3() {
}

void ParticleSystemData3::resize(size_t newNumberOfParticles) {
    _positions.resize(newNumberOfParticles, Vector3D());
    _velocities.resize(newNumberOfParticles, Vector3D());
    _forces.resize(newNumberOfParticles, Vector3D());

    for (auto& attr : _scalarDataList) {
        attr.resize(newNumberOfParticles, 0.0);
    }

    for (auto& attr : _vectorDataList) {
        attr.resize(newNumberOfParticles, Vector3D());
    }
}

size_t ParticleSystemData3::numberOfParticles() const {
    return _positions.size();
}

size_t ParticleSystemData3::addScalarData(double initialVal) {
    size_t attrIdx = _scalarDataList.size();
    _scalarDataList.emplace_back(numberOfParticles(), initialVal);
    return attrIdx;
}

size_t ParticleSystemData3::addVectorData(const Vector3D& initialVal) {
    size_t attrIdx = _vectorDataList.size();
    _vectorDataList.emplace_back(numberOfParticles(), initialVal);
    return attrIdx;
}

double ParticleSystemData3::radius() const {
    return _radius;
}

void ParticleSystemData3::setRadius(double newRadius) {
    _radius = std::max(newRadius, 0.0);
}

double ParticleSystemData3::mass() const {
    return _mass;
}

void ParticleSystemData3::setMass(double newMass) {
    _mass = std::max(newMass, 0.0);
}

ConstArrayAccessor1<Vector3D> ParticleSystemData3::positions() const {
    return _positions.constAccessor();
}

ArrayAccessor1<Vector3D> ParticleSystemData3::positions() {
    return _positions.accessor();
}

ConstArrayAccessor1<Vector3D> ParticleSystemData3::velocities() const {
    return _velocities.constAccessor();
}

ArrayAccessor1<Vector3D> ParticleSystemData3::velocities() {
    return _velocities.accessor();
}

ConstArrayAccessor1<Vector3D> ParticleSystemData3::forces() const {
    return _forces.constAccessor();
}

ArrayAccessor1<Vector3D> ParticleSystemData3::forces() {
    return _forces.accessor();
}

ConstArrayAccessor1<double> ParticleSystemData3::scalarDataAt(
    size_t idx) const {
    return _scalarDataList[idx].constAccessor();
}

ArrayAccessor1<double> ParticleSystemData3::scalarDataAt(size_t idx) {
    return _scalarDataList[idx].accessor();
}

ConstArrayAccessor1<Vector3D> ParticleSystemData3::vectorDataAt(
    size_t idx) const {
    return _vectorDataList[idx].constAccessor();
}

ArrayAccessor1<Vector3D> ParticleSystemData3::vectorDataAt(size_t idx) {
    return _vectorDataList[idx].accessor();
}

void ParticleSystemData3::addParticle(
    const Vector3D& newPosition,
    const Vector3D& newVelocity,
    const Vector3D& newForce) {
    Array1<Vector3D> newPositions = {newPosition};
    Array1<Vector3D> newVelocities = {newVelocity};
    Array1<Vector3D> newForces = {newForce};

    addParticles(
        newPositions.constAccessor(),
        newVelocities.constAccessor(),
        newForces.constAccessor());
}

void ParticleSystemData3::addParticles(
    const ConstArrayAccessor1<Vector3D>& newPositions,
    const ConstArrayAccessor1<Vector3D>& newVelocities,
    const ConstArrayAccessor1<Vector3D>& newForces) {
    JET_THROW_INVALID_ARG_IF(
        newVelocities.size() > 0
        && newVelocities.size() != newPositions.size());
    JET_THROW_INVALID_ARG_IF(
        newForces.size() > 0 && newForces.size() != newPositions.size());

    size_t oldNumberOfParticles = numberOfParticles();
    size_t newNumberOfParticles = oldNumberOfParticles + newPositions.size();

    resize(newNumberOfParticles);

    parallelFor(kZeroSize, newPositions.size(),
        [&](size_t i) {
            _positions[i + oldNumberOfParticles] = newPositions[i];
        });

    if (newVelocities.size() > 0) {
        parallelFor(kZeroSize, newPositions.size(),
            [&](size_t i) {
                _velocities[i + oldNumberOfParticles] = newVelocities[i];
            });
    }

    if (newForces.size() > 0) {
        parallelFor(kZeroSize, newPositions.size(),
            [&](size_t i) {
                _forces[i + oldNumberOfParticles] = newForces[i];
            });
    }
}

const PointNeighborSearcher3Ptr& ParticleSystemData3::neighborSearcher() const {
    return _neighborSearcher;
}

void ParticleSystemData3::setNeighborSearcher(
    const PointNeighborSearcher3Ptr& newNeighborSearcher) {
    _neighborSearcher = newNeighborSearcher;
}

const std::vector<std::vector<size_t>>&
ParticleSystemData3::neighborLists() const {
    return _neighborLists;
}

void ParticleSystemData3::buildNeighborSearcher(double maxSearchRadius) {
    Timer timer;

    // Use PointParallelHashGridSearcher3 by default
    _neighborSearcher = std::make_shared<PointParallelHashGridSearcher3>(
        kDefaultHashGridResolution,
        kDefaultHashGridResolution,
        kDefaultHashGridResolution,
        2.0 * maxSearchRadius);

    _neighborSearcher->build(positions());

    JET_INFO << "Building neighbor searcher took: "
             << timer.durationInSeconds()
             << " seconds";
}

void ParticleSystemData3::buildNeighborLists(double maxSearchRadius) {
    Timer timer;

    _neighborLists.resize(numberOfParticles());

    auto points = positions();
    for (size_t i = 0; i < numberOfParticles(); ++i) {
        Vector3D origin = points[i];
        _neighborLists[i].clear();

        _neighborSearcher->forEachNearbyPoint(
            origin,
            maxSearchRadius,
            [&](size_t j, const Vector3D&) {
                if (i != j) {
                    _neighborLists[i].push_back(j);
                }
            });
    }

    JET_INFO << "Building neighbor list took: "
             << timer.durationInSeconds()
             << " seconds";
}
