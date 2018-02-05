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
#include <generated/grid_system_data2_generated.h>

#include <jet/grid_system_data2.h>

#include <flatbuffers/flatbuffers.h>

#include <algorithm>
#include <vector>

using namespace jet;

GridSystemData2::GridSystemData2()
: GridSystemData2({0, 0}, {1, 1}, {0, 0}) {
}

GridSystemData2::GridSystemData2(
    const Size2& resolution,
    const Vector2D& gridSpacing,
    const Vector2D& origin) {
    _velocity = std::make_shared<FaceCenteredGrid2>();
    _advectableVectorDataList.push_back(_velocity);
    _velocityIdx = 0;
    resize(resolution, gridSpacing, origin);
}

GridSystemData2::GridSystemData2(const GridSystemData2& other) {
    resize(other._resolution, other._gridSpacing, other._origin);

    for (auto& data : other._scalarDataList) {
        _scalarDataList.push_back(data->clone());
    }
    for (auto& data : other._vectorDataList) {
        _vectorDataList.push_back(data->clone());
    }
    for (auto& data : other._advectableScalarDataList) {
        _advectableScalarDataList.push_back(data->clone());
    }
    for (auto& data : other._advectableVectorDataList) {
        _advectableVectorDataList.push_back(data->clone());
    }

    JET_ASSERT(_advectableVectorDataList.size() > 0);

    _velocity = std::dynamic_pointer_cast<FaceCenteredGrid2>(
        _advectableVectorDataList[0]);

    JET_ASSERT(_velocity != nullptr);

    _velocityIdx = 0;
}

GridSystemData2::~GridSystemData2() {
}

void GridSystemData2::resize(
    const Size2& resolution,
    const Vector2D& gridSpacing,
    const Vector2D& origin) {
    _resolution = resolution;
    _gridSpacing = gridSpacing;
    _origin = origin;

    for (auto& data : _scalarDataList) {
        data->resize(resolution, gridSpacing, origin);
    }
    for (auto& data : _vectorDataList) {
        data->resize(resolution, gridSpacing, origin);
    }
    for (auto& data : _advectableScalarDataList) {
        data->resize(resolution, gridSpacing, origin);
    }
    for (auto& data : _advectableVectorDataList) {
        data->resize(resolution, gridSpacing, origin);
    }
}

Size2 GridSystemData2::resolution() const {
    return _resolution;
}

Vector2D GridSystemData2::gridSpacing() const {
    return _gridSpacing;
}

Vector2D GridSystemData2::origin() const {
    return _origin;
}

BoundingBox2D GridSystemData2::boundingBox() const {
    return _velocity->boundingBox();
}

size_t GridSystemData2::addScalarData(
    const ScalarGridBuilder2Ptr& builder,
    double initialVal) {
    size_t attrIdx = _scalarDataList.size();
    _scalarDataList.push_back(
        builder->build(resolution(), gridSpacing(), origin(), initialVal));
    return attrIdx;
}

size_t GridSystemData2::addVectorData(
    const VectorGridBuilder2Ptr& builder,
    const Vector2D& initialVal) {
    size_t attrIdx = _vectorDataList.size();
    _vectorDataList.push_back(
        builder->build(resolution(), gridSpacing(), origin(), initialVal));
    return attrIdx;
}

size_t GridSystemData2::addAdvectableScalarData(
    const ScalarGridBuilder2Ptr& builder,
    double initialVal) {
    size_t attrIdx = _advectableScalarDataList.size();
    _advectableScalarDataList.push_back(
        builder->build(resolution(), gridSpacing(), origin(), initialVal));
    return attrIdx;
}

size_t GridSystemData2::addAdvectableVectorData(
    const VectorGridBuilder2Ptr& builder,
    const Vector2D& initialVal) {
    size_t attrIdx = _advectableVectorDataList.size();
    _advectableVectorDataList.push_back(
        builder->build(resolution(), gridSpacing(), origin(), initialVal));
    return attrIdx;
}

const FaceCenteredGrid2Ptr& GridSystemData2::velocity() const {
    return _velocity;
}

size_t GridSystemData2::velocityIndex() const {
    return _velocityIdx;
}

const ScalarGrid2Ptr& GridSystemData2::scalarDataAt(size_t idx) const {
    return _scalarDataList[idx];
}

const VectorGrid2Ptr& GridSystemData2::vectorDataAt(size_t idx) const {
    return _vectorDataList[idx];
}

const ScalarGrid2Ptr&
GridSystemData2::advectableScalarDataAt(size_t idx) const {
    return _advectableScalarDataList[idx];
}

const VectorGrid2Ptr&
GridSystemData2::advectableVectorDataAt(size_t idx) const {
    return _advectableVectorDataList[idx];
}

size_t GridSystemData2::numberOfScalarData() const {
    return _scalarDataList.size();
}

size_t GridSystemData2::numberOfVectorData() const {
    return _vectorDataList.size();
}

size_t GridSystemData2::numberOfAdvectableScalarData() const {
    return _advectableScalarDataList.size();
}

size_t GridSystemData2::numberOfAdvectableVectorData() const {
    return _advectableVectorDataList.size();
}

void GridSystemData2::serialize(std::vector<uint8_t>* buffer) const {
    flatbuffers::FlatBufferBuilder builder(1024);

    auto resolution = jetToFbs(_resolution);
    auto gridSpacing = jetToFbs(_gridSpacing);
    auto origin = jetToFbs(_origin);

    std::vector<flatbuffers::Offset<fbs::ScalarGridSerialized2>> scalarDataList;
    std::vector<flatbuffers::Offset<fbs::VectorGridSerialized2>> vectorDataList;
    std::vector<flatbuffers::Offset<fbs::ScalarGridSerialized2>>
        advScalarDataList;
    std::vector<flatbuffers::Offset<fbs::VectorGridSerialized2>>
        advVectorDataList;

    serializeGrid(
        &builder,
        _scalarDataList,
        fbs::CreateScalarGridSerialized2,
        &scalarDataList);
    serializeGrid(
        &builder,
        _vectorDataList,
        fbs::CreateVectorGridSerialized2,
        &vectorDataList);
    serializeGrid(
        &builder,
        _advectableScalarDataList,
        fbs::CreateScalarGridSerialized2,
        &advScalarDataList);
    serializeGrid(
        &builder,
        _advectableVectorDataList,
        fbs::CreateVectorGridSerialized2,
        &advVectorDataList);

    auto gsd = fbs::CreateGridSystemData2(
        builder,
        &resolution,
        &gridSpacing,
        &origin,
        _velocityIdx,
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

void GridSystemData2::deserialize(const std::vector<uint8_t>& buffer) {
    auto gsd = fbs::GetGridSystemData2(buffer.data());

    resize(
        fbsToJet(*gsd->resolution()),
        fbsToJet(*gsd->gridSpacing()),
        fbsToJet(*gsd->origin()));

    _scalarDataList.clear();
    _vectorDataList.clear();
    _advectableScalarDataList.clear();
    _advectableVectorDataList.clear();

    deserializeGrid(
        gsd->scalarData(),
        Factory::buildScalarGrid2,
        &_scalarDataList);
    deserializeGrid(
        gsd->vectorData(),
        Factory::buildVectorGrid2,
        &_vectorDataList);
    deserializeGrid(
        gsd->advectableScalarData(),
        Factory::buildScalarGrid2,
        &_advectableScalarDataList);
    deserializeGrid(
        gsd->advectableVectorData(),
        Factory::buildVectorGrid2,
        &_advectableVectorDataList);

    _velocityIdx = static_cast<size_t>(gsd->velocityIdx());
    _velocity = std::dynamic_pointer_cast<FaceCenteredGrid2>(
        _advectableVectorDataList[_velocityIdx]);
}
