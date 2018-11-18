// Copyright (c) 2018 Doyub Kim
//
// I am making my contributions/submissions to this project solely in my
// personal capacity and am not conveying any rights to any intellectual
// property of any third parties.

#ifdef _MSC_VER
#pragma warning(disable : 4244)
#endif

#include <pch.h>

#include <factory.h>
#include <fbs_helpers.h>
#include <generated/particle_system_data2_generated.h>
#include <generated/particle_system_data3_generated.h>

#include <jet/parallel.h>
#include <jet/particle_system_data.h>
#include <jet/point_parallel_hash_grid_searcher.h>
#include <jet/timer.h>

namespace jet {

static const size_t kDefaultHashGridResolution = 64;

// MARK: Serialization helpers

template <size_t N>
struct GetFlatbuffersParticleSystemData {};

template <>
struct GetFlatbuffersParticleSystemData<2> {
    using Offset = flatbuffers::Offset<fbs::ParticleSystemData2>;

    static const fbs::ParticleSystemData2 *getParticleSystemData(
        const uint8_t *data) {
        return fbs::GetParticleSystemData2(data);
    }
};

template <>
struct GetFlatbuffersParticleSystemData<3> {
    using Offset = flatbuffers::Offset<fbs::ParticleSystemData3>;

    static const fbs::ParticleSystemData3 *getParticleSystemData(
        const uint8_t *data) {
        return fbs::GetParticleSystemData3(data);
    }
};

// MARK: ParticleSystemData implementations

template <size_t N>
ParticleSystemData<N>::ParticleSystemData() : ParticleSystemData(0) {}

template <size_t N>
ParticleSystemData<N>::ParticleSystemData(size_t numberOfParticles) {
    _positionIdx = addVectorData();
    _velocityIdx = addVectorData();
    _forceIdx = addVectorData();

    // Use PointParallelHashGridSearcher<N> by default
    _neighborSearcher = std::make_shared<PointParallelHashGridSearcher<N>>(
        Vector<size_t, N>::makeConstant(kDefaultHashGridResolution),
        2.0 * _radius);

    resize(numberOfParticles);
}

template <size_t N>
ParticleSystemData<N>::ParticleSystemData(const ParticleSystemData &other) {
    set(other);
}

template <size_t N>
ParticleSystemData<N>::~ParticleSystemData() {}

template <size_t N>
void ParticleSystemData<N>::resize(size_t newNumberOfParticles) {
    _numberOfParticles = newNumberOfParticles;

    for (auto &attr : _scalarDataList) {
        attr.resize(newNumberOfParticles, 0.0);
    }

    for (auto &attr : _vectorDataList) {
        attr.resize(newNumberOfParticles, Vector<double, N>());
    }
}

template <size_t N>
size_t ParticleSystemData<N>::numberOfParticles() const {
    return _numberOfParticles;
}

template <size_t N>
size_t ParticleSystemData<N>::addScalarData(double initialVal) {
    size_t attrIdx = _scalarDataList.length();
    _scalarDataList.append(ScalarData(numberOfParticles(), initialVal));
    return attrIdx;
}

template <size_t N>
size_t ParticleSystemData<N>::addVectorData(
    const Vector<double, N> &initialVal) {
    size_t attrIdx = _vectorDataList.length();
    _vectorDataList.append(VectorData(numberOfParticles(), initialVal));
    return attrIdx;
}

template <size_t N>
double ParticleSystemData<N>::radius() const {
    return _radius;
}

template <size_t N>
void ParticleSystemData<N>::setRadius(double newRadius) {
    _radius = std::max(newRadius, 0.0);
}

template <size_t N>
double ParticleSystemData<N>::mass() const {
    return _mass;
}

template <size_t N>
void ParticleSystemData<N>::setMass(double newMass) {
    _mass = std::max(newMass, 0.0);
}

template <size_t N>
ConstArrayView1<Vector<double, N>> ParticleSystemData<N>::positions() const {
    return vectorDataAt(_positionIdx);
}

template <size_t N>
ArrayView1<Vector<double, N>> ParticleSystemData<N>::positions() {
    return vectorDataAt(_positionIdx);
}

template <size_t N>
ConstArrayView1<Vector<double, N>> ParticleSystemData<N>::velocities() const {
    return vectorDataAt(_velocityIdx);
}

template <size_t N>
ArrayView1<Vector<double, N>> ParticleSystemData<N>::velocities() {
    return vectorDataAt(_velocityIdx);
}

template <size_t N>
ConstArrayView1<Vector<double, N>> ParticleSystemData<N>::forces() const {
    return vectorDataAt(_forceIdx);
}

template <size_t N>
ArrayView1<Vector<double, N>> ParticleSystemData<N>::forces() {
    return vectorDataAt(_forceIdx);
}

template <size_t N>
ConstArrayView1<double> ParticleSystemData<N>::scalarDataAt(size_t idx) const {
    return ConstArrayView1<double>(_scalarDataList[idx]);
}

template <size_t N>
ArrayView1<double> ParticleSystemData<N>::scalarDataAt(size_t idx) {
    return ArrayView1<double>(_scalarDataList[idx]);
}

template <size_t N>
ConstArrayView1<Vector<double, N>> ParticleSystemData<N>::vectorDataAt(
    size_t idx) const {
    return ConstArrayView1<Vector<double, N>>(_vectorDataList[idx]);
}

template <size_t N>
ArrayView1<Vector<double, N>> ParticleSystemData<N>::vectorDataAt(size_t idx) {
    return ArrayView1<Vector<double, N>>(_vectorDataList[idx]);
}

template <size_t N>
void ParticleSystemData<N>::addParticle(const Vector<double, N> &newPosition,
                                        const Vector<double, N> &newVelocity,
                                        const Vector<double, N> &newForce) {
    Array1<Vector<double, N>> newPositions = {newPosition};
    Array1<Vector<double, N>> newVelocities = {newVelocity};
    Array1<Vector<double, N>> newForces = {newForce};

    addParticles(newPositions, newVelocities, newForces);
}

template <size_t N>
void ParticleSystemData<N>::addParticles(
    const ConstArrayView1<Vector<double, N>> &newPositions,
    const ConstArrayView1<Vector<double, N>> &newVelocities,
    const ConstArrayView1<Vector<double, N>> &newForces) {
    JET_THROW_INVALID_ARG_IF(newVelocities.length() > 0 &&
                             newVelocities.length() != newPositions.length());
    JET_THROW_INVALID_ARG_IF(newForces.length() > 0 &&
                             newForces.length() != newPositions.length());

    size_t oldNumberOfParticles = numberOfParticles();
    size_t newNumberOfParticles = oldNumberOfParticles + newPositions.length();

    resize(newNumberOfParticles);

    auto pos = positions();
    auto vel = velocities();
    auto frc = forces();

    parallelFor(kZeroSize, newPositions.length(), [&](size_t i) {
        pos[i + oldNumberOfParticles] = newPositions[i];
    });

    if (newVelocities.length() > 0) {
        parallelFor(kZeroSize, newPositions.length(), [&](size_t i) {
            vel[i + oldNumberOfParticles] = newVelocities[i];
        });
    }

    if (newForces.length() > 0) {
        parallelFor(kZeroSize, newPositions.length(), [&](size_t i) {
            frc[i + oldNumberOfParticles] = newForces[i];
        });
    }
}

template <size_t N>
const std::shared_ptr<PointNeighborSearcher<N>>
    &ParticleSystemData<N>::neighborSearcher() const {
    return _neighborSearcher;
}

template <size_t N>
void ParticleSystemData<N>::setNeighborSearcher(
    const std::shared_ptr<PointNeighborSearcher<N>> &newNeighborSearcher) {
    _neighborSearcher = newNeighborSearcher;
}

template <size_t N>
const Array1<Array1<size_t>> &ParticleSystemData<N>::neighborLists() const {
    return _neighborLists;
}

template <size_t N>
void ParticleSystemData<N>::buildNeighborSearcher(double maxSearchRadius) {
    Timer timer;

    JET_ASSERT(_neighborSearcher != nullptr);

    _neighborSearcher->build(positions(), maxSearchRadius);

    JET_INFO << "Building neighbor searcher took: " << timer.durationInSeconds()
             << " seconds";
}

template <size_t N>
void ParticleSystemData<N>::buildNeighborLists(double maxSearchRadius) {
    Timer timer;

    _neighborLists.resize(numberOfParticles());

    auto points = positions();
    for (size_t i = 0; i < numberOfParticles(); ++i) {
        Vector<double, N> origin = points[i];
        _neighborLists[i].clear();

        _neighborSearcher->forEachNearbyPoint(
            origin, maxSearchRadius, [&](size_t j, const Vector<double, N> &) {
                if (i != j) {
                    _neighborLists[i].append(j);
                }
            });
    }

    JET_INFO << "Building neighbor list took: " << timer.durationInSeconds()
             << " seconds";
}

template <size_t N>
void ParticleSystemData<N>::serialize(std::vector<uint8_t> *buffer) const {
    flatbuffers::FlatBufferBuilder builder(1024);
    typename GetFlatbuffersParticleSystemData<N>::Offset fbsParticleSystemData;

    serialize(*this, &builder, &fbsParticleSystemData);

    builder.Finish(fbsParticleSystemData);

    uint8_t *buf = builder.GetBufferPointer();
    size_t size = builder.GetSize();

    buffer->resize(size);
    memcpy(buffer->data(), buf, size);
}

template <size_t N>
void ParticleSystemData<N>::deserialize(const std::vector<uint8_t> &buffer) {
    auto fbsParticleSystemData =
        GetFlatbuffersParticleSystemData<N>::getParticleSystemData(
            buffer.data());
    deserialize(fbsParticleSystemData, *this);
}

template <size_t N>
void ParticleSystemData<N>::set(const ParticleSystemData &other) {
    _radius = other._radius;
    _mass = other._mass;
    _positionIdx = other._positionIdx;
    _velocityIdx = other._velocityIdx;
    _forceIdx = other._forceIdx;
    _numberOfParticles = other._numberOfParticles;

    for (auto &data : other._scalarDataList) {
        _scalarDataList.append(data);
    }

    for (auto &data : other._vectorDataList) {
        _vectorDataList.append(data);
    }

    _neighborSearcher = other._neighborSearcher->clone();
    _neighborLists = other._neighborLists;
}

template <size_t N>
ParticleSystemData<N> &ParticleSystemData<N>::operator=(
    const ParticleSystemData &other) {
    set(other);
    return *this;
}

template <size_t N>
template <size_t M>
std::enable_if_t<M == 2, void> ParticleSystemData<N>::serialize(
    const ParticleSystemData<2> &particles,
    flatbuffers::FlatBufferBuilder *builder,
    flatbuffers::Offset<fbs::ParticleSystemData2> *fbsParticleSystemData) {
    // Copy data
    std::vector<flatbuffers::Offset<fbs::ScalarParticleData2>> scalarDataList;
    for (const auto &scalarData : particles._scalarDataList) {
        auto fbsScalarData = fbs::CreateScalarParticleData2(
            *builder,
            builder->CreateVector(scalarData.data(), scalarData.length()));
        scalarDataList.push_back(fbsScalarData);
    }
    auto fbsScalarDataList = builder->CreateVector(scalarDataList);

    std::vector<flatbuffers::Offset<fbs::VectorParticleData2>> vectorDataList;
    for (const auto &vectorData : particles._vectorDataList) {
        std::vector<fbs::Vector2D> newVectorData;
        for (const auto &v : vectorData) {
            newVectorData.push_back(jetToFbs(v));
        }

        auto fbsVectorData = fbs::CreateVectorParticleData2(
            *builder, builder->CreateVectorOfStructs(newVectorData.data(),
                                                     newVectorData.size()));
        vectorDataList.push_back(fbsVectorData);
    }
    auto fbsVectorDataList = builder->CreateVector(vectorDataList);

    // Copy neighbor searcher
    auto neighborSearcherType =
        builder->CreateString(particles._neighborSearcher->typeName());
    std::vector<uint8_t> neighborSearcherSerialized;
    particles._neighborSearcher->serialize(&neighborSearcherSerialized);
    auto fbsNeighborSearcher = fbs::CreatePointNeighborSearcherSerialized2(
        *builder, neighborSearcherType,
        builder->CreateVector(neighborSearcherSerialized.data(),
                              neighborSearcherSerialized.size()));

    // Copy neighbor lists
    std::vector<flatbuffers::Offset<fbs::ParticleNeighborList2>> neighborLists;
    for (const auto &neighbors : particles._neighborLists) {
        std::vector<uint64_t> neighbors64(neighbors.begin(), neighbors.end());
        flatbuffers::Offset<fbs::ParticleNeighborList2> fbsNeighborList =
            fbs::CreateParticleNeighborList2(
                *builder,
                builder->CreateVector(neighbors64.data(), neighbors64.size()));
        neighborLists.push_back(fbsNeighborList);
    }

    auto fbsNeighborLists = builder->CreateVector(neighborLists);

    // Copy the searcher
    *fbsParticleSystemData = fbs::CreateParticleSystemData2(
        *builder, particles._radius, particles._mass, particles._positionIdx,
        particles._velocityIdx, particles._forceIdx, fbsScalarDataList,
        fbsVectorDataList, fbsNeighborSearcher, fbsNeighborLists);
}

template <size_t N>
template <size_t M>
std::enable_if_t<M == 3, void> ParticleSystemData<N>::serialize(
    const ParticleSystemData<3> &particles,
    flatbuffers::FlatBufferBuilder *builder,
    flatbuffers::Offset<fbs::ParticleSystemData3> *fbsParticleSystemData) {
    // Copy data
    std::vector<flatbuffers::Offset<fbs::ScalarParticleData3>> scalarDataList;
    for (const auto &scalarData : particles._scalarDataList) {
        auto fbsScalarData = fbs::CreateScalarParticleData3(
            *builder,
            builder->CreateVector(scalarData.data(), scalarData.length()));
        scalarDataList.push_back(fbsScalarData);
    }
    auto fbsScalarDataList = builder->CreateVector(scalarDataList);

    std::vector<flatbuffers::Offset<fbs::VectorParticleData3>> vectorDataList;
    for (const auto &vectorData : particles._vectorDataList) {
        std::vector<fbs::Vector3D> newVectorData;
        for (const auto &v : vectorData) {
            newVectorData.push_back(jetToFbs(v));
        }

        auto fbsVectorData = fbs::CreateVectorParticleData3(
            *builder, builder->CreateVectorOfStructs(newVectorData.data(),
                                                     newVectorData.size()));
        vectorDataList.push_back(fbsVectorData);
    }
    auto fbsVectorDataList = builder->CreateVector(vectorDataList);

    // Copy neighbor searcher
    auto neighborSearcherType =
        builder->CreateString(particles._neighborSearcher->typeName());
    std::vector<uint8_t> neighborSearcherSerialized;
    particles._neighborSearcher->serialize(&neighborSearcherSerialized);
    auto fbsNeighborSearcher = fbs::CreatePointNeighborSearcherSerialized3(
        *builder, neighborSearcherType,
        builder->CreateVector(neighborSearcherSerialized.data(),
                              neighborSearcherSerialized.size()));

    // Copy neighbor lists
    std::vector<flatbuffers::Offset<fbs::ParticleNeighborList3>> neighborLists;
    for (const auto &neighbors : particles._neighborLists) {
        std::vector<uint64_t> neighbors64(neighbors.begin(), neighbors.end());
        flatbuffers::Offset<fbs::ParticleNeighborList3> fbsNeighborList =
            fbs::CreateParticleNeighborList3(
                *builder,
                builder->CreateVector(neighbors64.data(), neighbors64.size()));
        neighborLists.push_back(fbsNeighborList);
    }

    auto fbsNeighborLists = builder->CreateVector(neighborLists);

    // Copy the searcher
    *fbsParticleSystemData = fbs::CreateParticleSystemData3(
        *builder, particles._radius, particles._mass, particles._positionIdx,
        particles._velocityIdx, particles._forceIdx, fbsScalarDataList,
        fbsVectorDataList, fbsNeighborSearcher, fbsNeighborLists);
}

template <size_t N>
template <size_t M>
std::enable_if_t<M == 2, void> ParticleSystemData<N>::deserialize(
    const fbs::ParticleSystemData2 *fbsParticleSystemData,
    ParticleSystemData<2> &particles) {
    particles._scalarDataList.clear();
    particles._vectorDataList.clear();

    // Copy scalars
    particles._radius = fbsParticleSystemData->radius();
    particles._mass = fbsParticleSystemData->mass();
    particles._positionIdx =
        static_cast<size_t>(fbsParticleSystemData->positionIdx());
    particles._velocityIdx =
        static_cast<size_t>(fbsParticleSystemData->velocityIdx());
    particles._forceIdx =
        static_cast<size_t>(fbsParticleSystemData->forceIdx());

    // Copy data
    auto fbsScalarDataList = fbsParticleSystemData->scalarDataList();
    for (const auto &fbsScalarData : (*fbsScalarDataList)) {
        auto data = fbsScalarData->data();

        particles._scalarDataList.append(ScalarData(data->size()));

        auto &newData = *(particles._scalarDataList.rbegin());

        for (uint32_t i = 0; i < data->size(); ++i) {
            newData[i] = data->Get(i);
        }
    }
    auto fbsVectorDataList = fbsParticleSystemData->vectorDataList();
    for (const auto &fbsVectorData : (*fbsVectorDataList)) {
        auto data = fbsVectorData->data();

        particles._vectorDataList.append(VectorData(data->size()));
        auto &newData = *(particles._vectorDataList.rbegin());
        for (uint32_t i = 0; i < data->size(); ++i) {
            newData[i] = fbsToJet(*data->Get(i));
        }
    }

    particles._numberOfParticles = particles._vectorDataList[0].length();

    // Copy neighbor searcher
    auto fbsNeighborSearcher = fbsParticleSystemData->neighborSearcher();
    particles._neighborSearcher = Factory::buildPointNeighborSearcher2(
        fbsNeighborSearcher->type()->c_str());
    std::vector<uint8_t> neighborSearcherSerialized(
        fbsNeighborSearcher->data()->begin(),
        fbsNeighborSearcher->data()->end());
    particles._neighborSearcher->deserialize(neighborSearcherSerialized);

    // Copy neighbor list
    auto fbsNeighborLists = fbsParticleSystemData->neighborLists();
    particles._neighborLists.resize(fbsNeighborLists->size());
    for (uint32_t i = 0; i < fbsNeighborLists->size(); ++i) {
        auto fbsNeighborList = fbsNeighborLists->Get(i);
        particles._neighborLists[i].resize(fbsNeighborList->data()->size());
        std::transform(fbsNeighborList->data()->begin(),
                       fbsNeighborList->data()->end(),
                       particles._neighborLists[i].begin(),
                       [](uint64_t val) { return static_cast<size_t>(val); });
    }
}

template <size_t N>
template <size_t M>
std::enable_if_t<M == 3, void> ParticleSystemData<N>::deserialize(
    const fbs::ParticleSystemData3 *fbsParticleSystemData,
    ParticleSystemData<3> &particles) {
    particles._scalarDataList.clear();
    particles._vectorDataList.clear();

    // Copy scalars
    particles._radius = fbsParticleSystemData->radius();
    particles._mass = fbsParticleSystemData->mass();
    particles._positionIdx =
        static_cast<size_t>(fbsParticleSystemData->positionIdx());
    particles._velocityIdx =
        static_cast<size_t>(fbsParticleSystemData->velocityIdx());
    particles._forceIdx =
        static_cast<size_t>(fbsParticleSystemData->forceIdx());

    // Copy data
    auto fbsScalarDataList = fbsParticleSystemData->scalarDataList();
    for (const auto &fbsScalarData : (*fbsScalarDataList)) {
        auto data = fbsScalarData->data();

        particles._scalarDataList.append(ScalarData(data->size()));

        auto &newData = *(particles._scalarDataList.rbegin());

        for (uint32_t i = 0; i < data->size(); ++i) {
            newData[i] = data->Get(i);
        }
    }

    auto fbsVectorDataList = fbsParticleSystemData->vectorDataList();
    for (const auto &fbsVectorData : (*fbsVectorDataList)) {
        auto data = fbsVectorData->data();

        particles._vectorDataList.append(VectorData(data->size()));
        auto &newData = *(particles._vectorDataList.rbegin());
        for (uint32_t i = 0; i < data->size(); ++i) {
            newData[i] = fbsToJet(*data->Get(i));
        }
    }

    particles._numberOfParticles = particles._vectorDataList[0].length();

    // Copy neighbor searcher
    auto fbsNeighborSearcher = fbsParticleSystemData->neighborSearcher();
    particles._neighborSearcher = Factory::buildPointNeighborSearcher3(
        fbsNeighborSearcher->type()->c_str());
    std::vector<uint8_t> neighborSearcherSerialized(
        fbsNeighborSearcher->data()->begin(),
        fbsNeighborSearcher->data()->end());
    particles._neighborSearcher->deserialize(neighborSearcherSerialized);

    // Copy neighbor list
    auto fbsNeighborLists = fbsParticleSystemData->neighborLists();
    particles._neighborLists.resize(fbsNeighborLists->size());
    for (uint32_t i = 0; i < fbsNeighborLists->size(); ++i) {
        auto fbsNeighborList = fbsNeighborLists->Get(i);
        particles._neighborLists[i].resize(fbsNeighborList->data()->size());
        std::transform(fbsNeighborList->data()->begin(),
                       fbsNeighborList->data()->end(),
                       particles._neighborLists[i].begin(),
                       [](uint64_t val) { return static_cast<size_t>(val); });
    }
}

template class ParticleSystemData<2>;

template class ParticleSystemData<3>;

}  // namespace jet
