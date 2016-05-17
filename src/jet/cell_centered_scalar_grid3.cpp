// Copyright (c) 2016 Doyub Kim

#include <pch.h>
#include <jet/cell_centered_scalar_grid3.h>

#include <algorithm>

using namespace jet;

CellCenteredScalarGrid3::CellCenteredScalarGrid3() {
}

CellCenteredScalarGrid3::CellCenteredScalarGrid3(
    size_t resolutionX,
    size_t resolutionY,
    size_t resolutionZ,
    double gridSpacingX,
    double gridSpacingY,
    double gridSpacingZ,
    double originX,
    double originY,
    double originZ,
    double initialValue) {
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
        initialValue);
}

CellCenteredScalarGrid3::CellCenteredScalarGrid3(
    const Size3& resolution,
    const Vector3D& gridSpacing,
    const Vector3D& origin,
    double initialValue) {
    resize(resolution, gridSpacing, origin, initialValue);
}

CellCenteredScalarGrid3::CellCenteredScalarGrid3(
    const CellCenteredScalarGrid3& other) {
    set(other);
}

Size3 CellCenteredScalarGrid3::dataSize() const {
    // The size of the data should be the same as the grid resolution.
    return resolution();
}

Vector3D CellCenteredScalarGrid3::dataOrigin() const {
    return origin() + 0.5 * gridSpacing();
}

std::shared_ptr<ScalarGrid3> CellCenteredScalarGrid3::clone() const {
    return std::make_shared<CellCenteredScalarGrid3>(*this);
}

void CellCenteredScalarGrid3::swap(Grid3* other) {
    CellCenteredScalarGrid3* sameType
        = dynamic_cast<CellCenteredScalarGrid3*>(other);
    if (sameType != nullptr) {
        swapScalarGrid(sameType);
    }
}

void CellCenteredScalarGrid3::set(const CellCenteredScalarGrid3& other) {
    setScalarGrid(other);
}

CellCenteredScalarGrid3&
CellCenteredScalarGrid3::operator=(const CellCenteredScalarGrid3& other) {
    set(other);
    return *this;
}

ScalarGridBuilder3Ptr CellCenteredScalarGrid3::builder() {
    return std::make_shared<CellCenteredScalarGridBuilder3>();
}


CellCenteredScalarGridBuilder3::CellCenteredScalarGridBuilder3() {
}

ScalarGrid3Ptr CellCenteredScalarGridBuilder3::build(
        const Size3& resolution,
        const Vector3D& gridSpacing,
        const Vector3D& gridOrigin,
        double initialVal) const {
    return std::make_shared<CellCenteredScalarGrid3>(
        resolution,
        gridSpacing,
        gridOrigin,
        initialVal);
}
