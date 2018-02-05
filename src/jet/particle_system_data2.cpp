// Copyright (c) 2018 Doyub Kim
//
// I am making my contributions/submissions to this project solely in my
// personal capacity and am not conveying any rights to any intellectual
// property of any third parties.

#ifdef _MSC_VER
#pragma warning(disable: 4244)
#endif

#include <pch.h>

#include <factory.h>
#include <fbs_helpers.h>
#include <generated/particle_system_data2_generated.h>

#include <jet/parallel.h>
#include <jet/particle_system_data2.h>
#include <jet/point_parallel_hash_grid_searcher2.h>
#include <jet/timer.h>

#include <algorithm>
#include <vector>

using namespace jet;

static const size_t kDefaultHashGridResolution = 64;

ParticleSystemData2::ParticleSystemData2()
: ParticleSystemData2(0) {
}

ParticleSystemData2::ParticleSystemData2(size_t numberOfParticles) {
    _positionIdx = addVectorData();
    _velocityIdx = addVectorData();
    _forceIdx = addVectorData();

    // Use PointParallelHashGridSearcher2 by default
    _neighborSearcher = std::make_shared<PointParallelHashGridSearcher2>(
        kDefaultHashGridResolution,
        kDefaultHashGridResolution,
        2.0 * _radius);

    resize(numberOfParticles);
}

ParticleSystemData2::ParticleSystemData2(const ParticleSystemData2& other) {
    set(other);
}

ParticleSystemData2::~ParticleSystemData2() {
}

void ParticleSystemData2::resize(size_t newNumberOfParticles) {
    _numberOfParticles = newNumberOfParticles;

    for (auto& attr : _scalarDataList) {
        attr.resize(newNumberOfParticles, 0.0);
    }

    for (auto& attr : _vectorDataList) {
        attr.resize(newNumberOfParticles, Vector2D());
    }
}

size_t ParticleSystemData2::numberOfParticles() const {
    return _numberOfParticles;
}

size_t ParticleSystemData2::addScalarData(double initialVal) {
    size_t attrIdx = _scalarDataList.size();
    _scalarDataList.emplace_back(numberOfParticles(), initialVal);
    return attrIdx;
}

size_t ParticleSystemData2::addVectorData(const Vector2D& initialVal) {
    size_t attrIdx = _vectorDataList.size();
    _vectorDataList.emplace_back(numberOfParticles(), initialVal);
    return attrIdx;
}

double ParticleSystemData2::radius() const {
    return _radius;
}

void ParticleSystemData2::setRadius(double newRadius) {
    _radius = std::max(newRadius, 0.0);
}

double ParticleSystemData2::mass() const {
    return _mass;
}

void ParticleSystemData2::setMass(double newMass) {
    _mass = std::max(newMass, 0.0);
}

ConstArrayAccessor1<Vector2D> ParticleSystemData2::positions() const {
    return vectorDataAt(_positionIdx);
}

ArrayAccessor1<Vector2D> ParticleSystemData2::positions() {
    return vectorDataAt(_positionIdx);
}

ConstArrayAccessor1<Vector2D> ParticleSystemData2::velocities() const {
    return vectorDataAt(_velocityIdx);
}

ArrayAccessor1<Vector2D> ParticleSystemData2::velocities() {
    return vectorDataAt(_velocityIdx);
}

ConstArrayAccessor1<Vector2D> ParticleSystemData2::forces() const {
    return vectorDataAt(_forceIdx);
}

ArrayAccessor1<Vector2D> ParticleSystemData2::forces() {
    return vectorDataAt(_forceIdx);
}

ConstArrayAccessor1<double> ParticleSystemData2::scalarDataAt(
    size_t idx) const {
    return _scalarDataList[idx].constAccessor();
}

ArrayAccessor1<double> ParticleSystemData2::scalarDataAt(size_t idx) {
    return _scalarDataList[idx].accessor();
}

ConstArrayAccessor1<Vector2D> ParticleSystemData2::vectorDataAt(
    size_t idx) const {
    return _vectorDataList[idx].constAccessor();
}

ArrayAccessor1<Vector2D> ParticleSystemData2::vectorDataAt(size_t idx) {
    return _vectorDataList[idx].accessor();
}

void ParticleSystemData2::addParticle(
    const Vector2D& newPosition,
    const Vector2D& newVelocity,
    const Vector2D& newForce) {
    Array1<Vector2D> newPositions = {newPosition};
    Array1<Vector2D> newVelocities = {newVelocity};
    Array1<Vector2D> newForces = {newForce};

    addParticles(
        newPositions.constAccessor(),
        newVelocities.constAccessor(),
        newForces.constAccessor());
}

void ParticleSystemData2::addParticles(
    const ConstArrayAccessor1<Vector2D>& newPositions,
    const ConstArrayAccessor1<Vector2D>& newVelocities,
    const ConstArrayAccessor1<Vector2D>& newForces) {
    JET_THROW_INVALID_ARG_IF(
        newVelocities.size() > 0
        && newVelocities.size() != newPositions.size());
    JET_THROW_INVALID_ARG_IF(
        newForces.size() > 0 && newForces.size() != newPositions.size());

    size_t oldNumberOfParticles = numberOfParticles();
    size_t newNumberOfParticles = oldNumberOfParticles + newPositions.size();

    resize(newNumberOfParticles);

    auto pos = positions();
    auto vel = velocities();
    auto frc = forces();

    parallelFor(kZeroSize, newPositions.size(),
        [&](size_t i) {
            pos[i + oldNumberOfParticles] = newPositions[i];
        });

    if (newVelocities.size() > 0) {
        parallelFor(kZeroSize, newPositions.size(),
            [&](size_t i) {
                vel[i + oldNumberOfParticles] = newVelocities[i];
            });
    }

    if (newForces.size() > 0) {
        parallelFor(kZeroSize, newPositions.size(),
            [&](size_t i) {
                frc[i + oldNumberOfParticles] = newForces[i];
            });
    }
}

const PointNeighborSearcher2Ptr& ParticleSystemData2::neighborSearcher() const {
    return _neighborSearcher;
}

void ParticleSystemData2::setNeighborSearcher(
    const PointNeighborSearcher2Ptr& newNeighborSearcher) {
    _neighborSearcher = newNeighborSearcher;
}

const std::vector<std::vector<size_t>>&
ParticleSystemData2::neighborLists() const {
    return _neighborLists;
}

void ParticleSystemData2::buildNeighborSearcher(double maxSearchRadius) {
    Timer timer;

    // Use PointParallelHashGridSearcher2 by default
    _neighborSearcher = std::make_shared<PointParallelHashGridSearcher2>(
        kDefaultHashGridResolution,
        kDefaultHashGridResolution,
        2.0 * maxSearchRadius);

    _neighborSearcher->build(positions());

    JET_INFO << "Building neighbor searcher took: "
             << timer.durationInSeconds()
             << " seconds";
}

void ParticleSystemData2::buildNeighborLists(double maxSearchRadius) {
    Timer timer;

    _neighborLists.resize(numberOfParticles());

    auto points = positions();
    for (size_t i = 0; i < numberOfParticles(); ++i) {
        Vector2D origin = points[i];
        _neighborLists[i].clear();

        _neighborSearcher->forEachNearbyPoint(
            origin,
            maxSearchRadius,
            [&](size_t j, const Vector2D&) {
                if (i != j) {
                    _neighborLists[i].push_back(j);
                }
            });
    }

    JET_INFO << "Building neighbor list took: "
             << timer.durationInSeconds()
             << " seconds";
}

void ParticleSystemData2::serialize(std::vector<uint8_t>* buffer) const {
    flatbuffers::FlatBufferBuilder builder(1024);
    flatbuffers::Offset<fbs::ParticleSystemData2> fbsParticleSystemData;

    serializeParticleSystemData(&builder, &fbsParticleSystemData);

    builder.Finish(fbsParticleSystemData);

    uint8_t *buf = builder.GetBufferPointer();
    size_t size = builder.GetSize();

    buffer->resize(size);
    memcpy(buffer->data(), buf, size);
}

void ParticleSystemData2::deserialize(const std::vector<uint8_t>& buffer) {
    auto fbsParticleSystemData = fbs::GetParticleSystemData2(buffer.data());
    deserializeParticleSystemData(fbsParticleSystemData);
}

void ParticleSystemData2::set(const ParticleSystemData2& other) {
    _radius = other._radius;
    _mass = other._mass;
    _positionIdx = other._positionIdx;
    _velocityIdx = other._velocityIdx;
    _forceIdx = other._forceIdx;
    _numberOfParticles = other._numberOfParticles;

    for (auto& attr : other._scalarDataList) {
        _scalarDataList.emplace_back(attr);
    }

    for (auto& attr : other._vectorDataList) {
        _vectorDataList.emplace_back(attr);
    }

    _neighborSearcher = other._neighborSearcher->clone();
    _neighborLists = other._neighborLists;
}

ParticleSystemData2& ParticleSystemData2::operator=(
    const ParticleSystemData2& other) {
    set(other);
    return *this;
}

void ParticleSystemData2::serializeParticleSystemData(
    flatbuffers::FlatBufferBuilder* builder,
    flatbuffers::Offset<fbs::ParticleSystemData2>* fbsParticleSystemData)
    const {
    // Copy data
    std::vector<flatbuffers::Offset<fbs::ScalarParticleData2>> scalarDataList;
    for (const auto& scalarData : _scalarDataList) {
        auto fbsScalarData = fbs::CreateScalarParticleData2(
                *builder,
                builder->CreateVector(scalarData.data(), scalarData.size()));
        scalarDataList.push_back(fbsScalarData);
    }
    auto fbsScalarDataList = builder->CreateVector(scalarDataList);

    std::vector<flatbuffers::Offset<fbs::VectorParticleData2>> vectorDataList;
    for (const auto& vectorData : _vectorDataList) {
        std::vector<fbs::Vector2D> newVectorData;
        for (const auto& v : vectorData) {
            newVectorData.push_back(jetToFbs(v));
        }

        auto fbsVectorData = fbs::CreateVectorParticleData2(
                *builder,
                builder->CreateVectorOfStructs(
                    newVectorData.data(), newVectorData.size()));
        vectorDataList.push_back(fbsVectorData);
    }
    auto fbsVectorDataList = builder->CreateVector(vectorDataList);

    // Copy neighbor searcher
    auto neighborSearcherType
        = builder->CreateString(_neighborSearcher->typeName());
    std::vector<uint8_t> neighborSearcherSerialized;
    _neighborSearcher->serialize(&neighborSearcherSerialized);
    auto fbsNeighborSearcher = fbs::CreatePointNeighborSearcherSerialized2(
        *builder,
        neighborSearcherType,
        builder->CreateVector(
            neighborSearcherSerialized.data(),
            neighborSearcherSerialized.size()));

    // Copy neighbor lists
    std::vector<flatbuffers::Offset<fbs::ParticleNeighborList2>> neighborLists;
    for (const auto& neighbors : _neighborLists) {
        std::vector<uint64_t> neighbors64(neighbors.begin(), neighbors.end());
        flatbuffers::Offset<fbs::ParticleNeighborList2> fbsNeighborList
            = fbs::CreateParticleNeighborList2(
                *builder,
                builder->CreateVector(neighbors64.data(), neighbors64.size()));
        neighborLists.push_back(fbsNeighborList);
    }

    auto fbsNeighborLists = builder->CreateVector(neighborLists);

    // Copy the searcher
    *fbsParticleSystemData = fbs::CreateParticleSystemData2(
        *builder,
        _radius,
        _mass,
        _positionIdx,
        _velocityIdx,
        _forceIdx,
        fbsScalarDataList,
        fbsVectorDataList,
        fbsNeighborSearcher,
        fbsNeighborLists);
}

void ParticleSystemData2::deserializeParticleSystemData(
    const fbs::ParticleSystemData2* fbsParticleSystemData) {
    _scalarDataList.clear();
    _vectorDataList.clear();

    // Copy scalars
    _radius = fbsParticleSystemData->radius();
    _mass = fbsParticleSystemData->mass();
    _positionIdx = static_cast<size_t>(fbsParticleSystemData->positionIdx());
    _velocityIdx = static_cast<size_t>(fbsParticleSystemData->velocityIdx());
    _forceIdx = static_cast<size_t>(fbsParticleSystemData->forceIdx());

    // Copy data
    auto fbsScalarDataList = fbsParticleSystemData->scalarDataList();
    for (const auto& fbsScalarData : (*fbsScalarDataList)) {
        auto data = fbsScalarData->data();

        _scalarDataList.push_back(ScalarData(data->size()));

        auto& newData = *(_scalarDataList.rbegin());

        for (uint32_t i = 0; i < data->size(); ++i) {
            newData[i] = data->Get(i);
        }
    }

    auto fbsVectorDataList = fbsParticleSystemData->vectorDataList();
    for (const auto& fbsVectorData : (*fbsVectorDataList)) {
        auto data = fbsVectorData->data();

        _vectorDataList.push_back(VectorData(data->size()));
        auto& newData = *(_vectorDataList.rbegin());
        for (uint32_t i = 0; i < data->size(); ++i) {
            newData[i] = fbsToJet(*data->Get(i));
        }
    }

    _numberOfParticles = _vectorDataList[0].size();

    // Copy neighbor searcher
    auto fbsNeighborSearcher = fbsParticleSystemData->neighborSearcher();
    _neighborSearcher
        = Factory::buildPointNeighborSearcher2(
            fbsNeighborSearcher->type()->c_str());
    std::vector<uint8_t> neighborSearcherSerialized(
        fbsNeighborSearcher->data()->begin(),
        fbsNeighborSearcher->data()->end());
    _neighborSearcher->deserialize(neighborSearcherSerialized);

    // Copy neighbor list
    auto fbsNeighborLists = fbsParticleSystemData->neighborLists();
    _neighborLists.resize(fbsNeighborLists->size());
    for (uint32_t i = 0; i < fbsNeighborLists->size(); ++i) {
        auto fbsNeighborList = fbsNeighborLists->Get(i);
        _neighborLists[i].resize(fbsNeighborList->data()->size());
        std::transform(
            fbsNeighborList->data()->begin(),
            fbsNeighborList->data()->end(),
            _neighborLists[i].begin(),
            [] (uint64_t val) {
                return static_cast<size_t>(val);
            });
    }
}
