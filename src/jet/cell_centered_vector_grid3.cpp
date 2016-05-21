// Copyright (c) 2016 Doyub Kim

#include <pch.h>
#include <jet/cell_centered_vector_grid3.h>
#include <jet/parallel.h>
#include <algorithm>  // just make cpplint happy..

using namespace jet;

CellCenteredVectorGrid3::CellCenteredVectorGrid3() {
}

CellCenteredVectorGrid3::CellCenteredVectorGrid3(
    size_t resolutionX,
    size_t resolutionY,
    size_t resolutionZ,
    double gridSpacingX,
    double gridSpacingY,
    double gridSpacingZ,
    double originX,
    double originY,
    double originZ,
    double initialValueU,
    double initialValueV,
    double initialValueW) {
    resize(
        resolutionX,
        resolutionY,
        resolutionZ,
        gridSpacingX,
        gridSpacingY,
        gridSpacingZ,
        originX,
        originY,
        originZ,
        initialValueU,
        initialValueV,
        initialValueW);
}

CellCenteredVectorGrid3::CellCenteredVectorGrid3(
    const Size3& resolution,
    const Vector3D& gridSpacing,
    const Vector3D& origin,
    const Vector3D& initialValue) {
    resize(resolution, gridSpacing, origin, initialValue);
}

Size3 CellCenteredVectorGrid3::dataSize() const {
    return resolution();
}

Vector3D CellCenteredVectorGrid3::dataOrigin() const {
    return origin() + 0.5 * gridSpacing();
}

void CellCenteredVectorGrid3::swap(Grid3* other) {
    CellCenteredVectorGrid3* sameType
        = dynamic_cast<CellCenteredVectorGrid3*>(other);
    if (sameType != nullptr) {
        swapCollocatedVectorGrid(sameType);
    }
}

void CellCenteredVectorGrid3::fill(const Vector3D& value) {
    Size3 size = dataSize();
    auto acc = dataAccessor();
    parallelFor(
        kZeroSize, size.x,
        kZeroSize, size.y,
        kZeroSize, size.z,
        [this, value, &acc](size_t i, size_t j, size_t k) {
            acc(i, j, k) = value;
        });
}

void CellCenteredVectorGrid3::fill(const std::function<Vector3D(
    const Vector3D&)>& func) {
    Size3 size = dataSize();
    auto acc = dataAccessor();
    DataPositionFunc pos = dataPosition();
    parallelFor(
        kZeroSize, size.x,
        kZeroSize, size.y,
        kZeroSize, size.z,
        [this, &func, &acc, &pos](size_t i, size_t j, size_t k) {
            acc(i, j, k) = func(pos(i, j, k));
        });
}

std::shared_ptr<VectorGrid3> CellCenteredVectorGrid3::clone() const {
    return std::make_shared<CellCenteredVectorGrid3>(*this);
}

VectorGridBuilder3Ptr CellCenteredVectorGrid3::builder() {
    return std::make_shared<CellCenteredVectorGridBuilder3>();
}


CellCenteredVectorGridBuilder3::CellCenteredVectorGridBuilder3() {
}

VectorGrid3Ptr CellCenteredVectorGridBuilder3::build(
    const Size3& resolution,
    const Vector3D& gridSpacing,
    const Vector3D& gridOrigin,
    const Vector3D& initialVal) const {
    return std::make_shared<CellCenteredVectorGrid3>(
        resolution,
        gridSpacing,
        gridOrigin,
        initialVal);
}
