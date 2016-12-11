// Copyright (c) 2016 Doyub Kim

#include <pch.h>
#include <jet/grid_system_data3.h>

using namespace jet;

GridSystemData3::GridSystemData3() {
    _velocity = std::make_shared<FaceCenteredGrid3>();
    _advectableVectorDataList.push_back(_velocity);
    _velocityIdx = 0;
}

GridSystemData3::~GridSystemData3() {
}

void GridSystemData3::resize(
    const Size3& resolution,
    const Vector3D& gridSpacing,
    const Vector3D& origin) {
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

Size3 GridSystemData3::resolution() const {
    return _velocity->resolution();
}

Vector3D GridSystemData3::gridSpacing() const {
    return _velocity->gridSpacing();
}

Vector3D GridSystemData3::origin() const {
    return _velocity->origin();
}

BoundingBox3D GridSystemData3::boundingBox() const {
    return _velocity->boundingBox();
}

size_t GridSystemData3::addScalarData(
    const ScalarGridBuilder3Ptr& builder,
    double initialVal) {
    size_t attrIdx = _scalarDataList.size();
    _scalarDataList.push_back(
        builder->build(resolution(), gridSpacing(), origin(), initialVal));
    return attrIdx;
}

size_t GridSystemData3::addVectorData(
    const VectorGridBuilder3Ptr& builder,
    const Vector3D& initialVal) {
    size_t attrIdx = _scalarDataList.size();
    _vectorDataList.push_back(
        builder->build(resolution(), gridSpacing(), origin(), initialVal));
    return attrIdx;
}

size_t GridSystemData3::addAdvectableScalarData(
    const ScalarGridBuilder3Ptr& builder,
    double initialVal) {
    size_t attrIdx = _advectableScalarDataList.size();
    _advectableScalarDataList.push_back(
        builder->build(resolution(), gridSpacing(), origin(), initialVal));
    return attrIdx;
}

size_t GridSystemData3::addAdvectableVectorData(
    const VectorGridBuilder3Ptr& builder,
    const Vector3D& initialVal) {
    size_t attrIdx = _advectableVectorDataList.size();
    _advectableVectorDataList.push_back(
        builder->build(resolution(), gridSpacing(), origin(), initialVal));
    return attrIdx;
}

const FaceCenteredGrid3Ptr& GridSystemData3::velocity() const {
    return _velocity;
}

size_t GridSystemData3::velocityIndex() const {
    return _velocityIdx;
}

const ScalarGrid3Ptr& GridSystemData3::scalarDataAt(size_t idx) const {
    return _scalarDataList[idx];
}

const VectorGrid3Ptr& GridSystemData3::vectorDataAt(size_t idx) const {
    return _vectorDataList[idx];
}

const ScalarGrid3Ptr&
GridSystemData3::advectableScalarDataAt(size_t idx) const {
    return _advectableScalarDataList[idx];
}

const VectorGrid3Ptr&
GridSystemData3::advectableVectorDataAt(size_t idx) const {
    return _advectableVectorDataList[idx];
}

size_t GridSystemData3::numberOfScalarData() const {
    return _scalarDataList.size();
}

size_t GridSystemData3::numberOfVectorData() const {
    return _vectorDataList.size();
}

size_t GridSystemData3::numberOfAdvectableScalarData() const {
    return _advectableScalarDataList.size();
}

size_t GridSystemData3::numberOfAdvectableVectorData() const {
    return _advectableVectorDataList.size();
}
