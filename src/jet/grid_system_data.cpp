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
#include <generated/grid_system_data2_generated.h>
#include <generated/grid_system_data3_generated.h>

#include <jet/grid_system_data.h>

#include <flatbuffers/flatbuffers.h>

namespace jet {

template <size_t N>
GridSystemData<N>::GridSystemData()
    : GridSystemData(Vector<size_t, N>(), Vector<double, N>::makeConstant(1.0),
                     Vector<double, N>()) {}

template <size_t N>
GridSystemData<N>::GridSystemData(const Vector<size_t, N> &resolution,
                                  const Vector<double, N> &gridSpacing,
                                  const Vector<double, N> &origin) {
    _velocity = std::make_shared<FaceCenteredGrid<N>>();
    _advectableVectorDataList.push_back(_velocity);
    _velocityIdx = 0;
    resize(resolution, gridSpacing, origin);
}

template <size_t N>
GridSystemData<N>::GridSystemData(const GridSystemData &other) {
    resize(other._resolution, other._gridSpacing, other._origin);

    for (auto &data : other._scalarDataList) {
        _scalarDataList.push_back(data->clone());
    }
    for (auto &data : other._vectorDataList) {
        _vectorDataList.push_back(data->clone());
    }
    for (auto &data : other._advectableScalarDataList) {
        _advectableScalarDataList.push_back(data->clone());
    }
    for (auto &data : other._advectableVectorDataList) {
        _advectableVectorDataList.push_back(data->clone());
    }

    JET_ASSERT(_advectableVectorDataList.size() > 0);

    _velocity = std::dynamic_pointer_cast<FaceCenteredGrid<N>>(
        _advectableVectorDataList[0]);

    JET_ASSERT(_velocity != nullptr);

    _velocityIdx = 0;
}

template <size_t N>
GridSystemData<N>::~GridSystemData() {}

template <size_t N>
void GridSystemData<N>::resize(const Vector<size_t, N> &resolution,
                               const Vector<double, N> &gridSpacing,
                               const Vector<double, N> &origin) {
    _resolution = resolution;
    _gridSpacing = gridSpacing;
    _origin = origin;

    for (auto &data : _scalarDataList) {
        data->resize(resolution, gridSpacing, origin);
    }
    for (auto &data : _vectorDataList) {
        data->resize(resolution, gridSpacing, origin);
    }
    for (auto &data : _advectableScalarDataList) {
        data->resize(resolution, gridSpacing, origin);
    }
    for (auto &data : _advectableVectorDataList) {
        data->resize(resolution, gridSpacing, origin);
    }
}

template <size_t N>
Vector<size_t, N> GridSystemData<N>::resolution() const {
    return _resolution;
}

template <size_t N>
Vector<double, N> GridSystemData<N>::gridSpacing() const {
    return _gridSpacing;
}

template <size_t N>
Vector<double, N> GridSystemData<N>::origin() const {
    return _origin;
}

template <size_t N>
BoundingBox<double, N> GridSystemData<N>::boundingBox() const {
    return _velocity->boundingBox();
}

template <size_t N>
size_t GridSystemData<N>::addScalarData(
    const std::shared_ptr<ScalarGridBuilder<N>> &builder, double initialVal) {
    size_t attrIdx = _scalarDataList.size();
    _scalarDataList.push_back(
        builder->build(resolution(), gridSpacing(), origin(), initialVal));
    return attrIdx;
}

template <size_t N>
size_t GridSystemData<N>::addVectorData(
    const std::shared_ptr<VectorGridBuilder<N>> &builder,
    const Vector<double, N> &initialVal) {
    size_t attrIdx = _vectorDataList.size();
    _vectorDataList.push_back(
        builder->build(resolution(), gridSpacing(), origin(), initialVal));
    return attrIdx;
}

template <size_t N>
size_t GridSystemData<N>::addAdvectableScalarData(
    const std::shared_ptr<ScalarGridBuilder<N>> &builder, double initialVal) {
    size_t attrIdx = _advectableScalarDataList.size();
    _advectableScalarDataList.push_back(
        builder->build(resolution(), gridSpacing(), origin(), initialVal));
    return attrIdx;
}

template <size_t N>
size_t GridSystemData<N>::addAdvectableVectorData(
    const std::shared_ptr<VectorGridBuilder<N>> &builder,
    const Vector<double, N> &initialVal) {
    size_t attrIdx = _advectableVectorDataList.size();
    _advectableVectorDataList.push_back(
        builder->build(resolution(), gridSpacing(), origin(), initialVal));
    return attrIdx;
}

template <size_t N>
const std::shared_ptr<FaceCenteredGrid<N>> &GridSystemData<N>::velocity()
    const {
    return _velocity;
}

template <size_t N>
size_t GridSystemData<N>::velocityIndex() const {
    return _velocityIdx;
}

template <size_t N>
const std::shared_ptr<ScalarGrid<N>> &GridSystemData<N>::scalarDataAt(
    size_t idx) const {
    return _scalarDataList[idx];
}

template <size_t N>
const std::shared_ptr<VectorGrid<N>> &GridSystemData<N>::vectorDataAt(
    size_t idx) const {
    return _vectorDataList[idx];
}

template <size_t N>
const std::shared_ptr<ScalarGrid<N>> &GridSystemData<N>::advectableScalarDataAt(
    size_t idx) const {
    return _advectableScalarDataList[idx];
}

template <size_t N>
const std::shared_ptr<VectorGrid<N>> &GridSystemData<N>::advectableVectorDataAt(
    size_t idx) const {
    return _advectableVectorDataList[idx];
}

template <size_t N>
size_t GridSystemData<N>::numberOfScalarData() const {
    return _scalarDataList.size();
}

template <size_t N>
size_t GridSystemData<N>::numberOfVectorData() const {
    return _vectorDataList.size();
}

template <size_t N>
size_t GridSystemData<N>::numberOfAdvectableScalarData() const {
    return _advectableScalarDataList.size();
}

template <size_t N>
size_t GridSystemData<N>::numberOfAdvectableVectorData() const {
    return _advectableVectorDataList.size();
}

template <size_t N>
void GridSystemData<N>::serialize(std::vector<uint8_t> *buffer) const {
    serialize(*this, buffer);
}

template <size_t N>
void GridSystemData<N>::deserialize(const std::vector<uint8_t> &buffer) {
    deserialize(buffer, *this);
}

template <size_t N>
template <size_t M>
std::enable_if_t<M == 2, void> GridSystemData<N>::serialize(
    const GridSystemData<2> &grid, std::vector<uint8_t> *buffer) {
    flatbuffers::FlatBufferBuilder builder(1024);

    auto resolution = jetToFbs(grid._resolution);
    auto gridSpacing = jetToFbs(grid._gridSpacing);
    auto origin = jetToFbs(grid._origin);

    std::vector<flatbuffers::Offset<fbs::ScalarGridSerialized2>> scalarDataList;
    std::vector<flatbuffers::Offset<fbs::VectorGridSerialized2>> vectorDataList;
    std::vector<flatbuffers::Offset<fbs::ScalarGridSerialized2>>
        advScalarDataList;
    std::vector<flatbuffers::Offset<fbs::VectorGridSerialized2>>
        advVectorDataList;

    serializeGrid(&builder, grid._scalarDataList,
                  fbs::CreateScalarGridSerialized2, &scalarDataList);
    serializeGrid(&builder, grid._vectorDataList,
                  fbs::CreateVectorGridSerialized2, &vectorDataList);
    serializeGrid(&builder, grid._advectableScalarDataList,
                  fbs::CreateScalarGridSerialized2, &advScalarDataList);
    serializeGrid(&builder, grid._advectableVectorDataList,
                  fbs::CreateVectorGridSerialized2, &advVectorDataList);

    auto gsd = fbs::CreateGridSystemData2(
        builder, &resolution, &gridSpacing, &origin, grid._velocityIdx,
        builder.CreateVector(scalarDataList),
        builder.CreateVector(vectorDataList),
        builder.CreateVector(advScalarDataList),
        builder.CreateVector(advVectorDataList));

    builder.Finish(gsd);

    uint8_t *buf = builder.GetBufferPointer();
    size_t size = builder.GetSize();

    buffer->resize(size);
    memcpy(buffer->data(), buf, size);
}

template <size_t N>
template <size_t M>
std::enable_if_t<M == 3, void> GridSystemData<N>::serialize(
    const GridSystemData<3> &grid, std::vector<uint8_t> *buffer) {
    flatbuffers::FlatBufferBuilder builder(1024);

    auto resolution = jetToFbs(grid._resolution);
    auto gridSpacing = jetToFbs(grid._gridSpacing);
    auto origin = jetToFbs(grid._origin);

    std::vector<flatbuffers::Offset<fbs::ScalarGridSerialized3>> scalarDataList;
    std::vector<flatbuffers::Offset<fbs::VectorGridSerialized3>> vectorDataList;
    std::vector<flatbuffers::Offset<fbs::ScalarGridSerialized3>>
        advScalarDataList;
    std::vector<flatbuffers::Offset<fbs::VectorGridSerialized3>>
        advVectorDataList;

    serializeGrid(&builder, grid._scalarDataList,
                  fbs::CreateScalarGridSerialized3, &scalarDataList);
    serializeGrid(&builder, grid._vectorDataList,
                  fbs::CreateVectorGridSerialized3, &vectorDataList);
    serializeGrid(&builder, grid._advectableScalarDataList,
                  fbs::CreateScalarGridSerialized3, &advScalarDataList);
    serializeGrid(&builder, grid._advectableVectorDataList,
                  fbs::CreateVectorGridSerialized3, &advVectorDataList);

    auto gsd = fbs::CreateGridSystemData3(
        builder, &resolution, &gridSpacing, &origin, grid._velocityIdx,
        builder.CreateVector(scalarDataList),
        builder.CreateVector(vectorDataList),
        builder.CreateVector(advScalarDataList),
        builder.CreateVector(advVectorDataList));

    builder.Finish(gsd);

    uint8_t *buf = builder.GetBufferPointer();
    size_t size = builder.GetSize();

    buffer->resize(size);
    memcpy(buffer->data(), buf, size);
}

template <size_t N>
template <size_t M>
std::enable_if_t<M == 2, void> GridSystemData<N>::deserialize(
    const std::vector<uint8_t> &buffer, GridSystemData<2> &grid) {
    auto gsd = fbs::GetGridSystemData2(buffer.data());

    grid.resize(fbsToJet(*gsd->resolution()), fbsToJet(*gsd->gridSpacing()),
                fbsToJet(*gsd->origin()));

    grid._scalarDataList.clear();
    grid._vectorDataList.clear();
    grid._advectableScalarDataList.clear();
    grid._advectableVectorDataList.clear();

    deserializeGrid(gsd->scalarData(), Factory::buildScalarGrid2,
                    &grid._scalarDataList);
    deserializeGrid(gsd->vectorData(), Factory::buildVectorGrid2,
                    &grid._vectorDataList);
    deserializeGrid(gsd->advectableScalarData(), Factory::buildScalarGrid2,
                    &grid._advectableScalarDataList);
    deserializeGrid(gsd->advectableVectorData(), Factory::buildVectorGrid2,
                    &grid._advectableVectorDataList);

    grid._velocityIdx = static_cast<size_t>(gsd->velocityIdx());
    grid._velocity = std::dynamic_pointer_cast<FaceCenteredGrid2>(
        grid._advectableVectorDataList[grid._velocityIdx]);
}

template <size_t N>
template <size_t M>
std::enable_if_t<M == 3, void> GridSystemData<N>::deserialize(
    const std::vector<uint8_t> &buffer, GridSystemData<3> &grid) {
    auto gsd = fbs::GetGridSystemData3(buffer.data());

    grid.resize(fbsToJet(*gsd->resolution()), fbsToJet(*gsd->gridSpacing()),
                fbsToJet(*gsd->origin()));

    grid._scalarDataList.clear();
    grid._vectorDataList.clear();
    grid._advectableScalarDataList.clear();
    grid._advectableVectorDataList.clear();

    deserializeGrid(gsd->scalarData(), Factory::buildScalarGrid3,
                    &grid._scalarDataList);
    deserializeGrid(gsd->vectorData(), Factory::buildVectorGrid3,
                    &grid._vectorDataList);
    deserializeGrid(gsd->advectableScalarData(), Factory::buildScalarGrid3,
                    &grid._advectableScalarDataList);
    deserializeGrid(gsd->advectableVectorData(), Factory::buildVectorGrid3,
                    &grid._advectableVectorDataList);

    grid._velocityIdx = static_cast<size_t>(gsd->velocityIdx());
    grid._velocity = std::dynamic_pointer_cast<FaceCenteredGrid3>(
        grid._advectableVectorDataList[grid._velocityIdx]);
}

template class GridSystemData<2>;

template class GridSystemData<3>;

}  // namespace jet
