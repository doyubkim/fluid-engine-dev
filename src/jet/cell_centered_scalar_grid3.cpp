// Copyright (c) 2018 Doyub Kim
//
// I am making my contributions/submissions to this project solely in my
// personal capacity and am not conveying any rights to any intellectual
// property of any third parties.

#include <pch.h>
#include <jet/cell_centered_scalar_grid3.h>

#include <algorithm>
#include <utility>  // just make cpplint happy..

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
    return CLONE_W_CUSTOM_DELETER(CellCenteredScalarGrid3);
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

CellCenteredScalarGrid3::Builder CellCenteredScalarGrid3::builder() {
    return Builder();
}


CellCenteredScalarGrid3::Builder&
CellCenteredScalarGrid3::Builder::withResolution(const Size3& resolution) {
    _resolution = resolution;
    return *this;
}

CellCenteredScalarGrid3::Builder&
CellCenteredScalarGrid3::Builder::withResolution(
    size_t resolutionX, size_t resolutionY, size_t resolutionZ) {
    _resolution.x = resolutionX;
    _resolution.y = resolutionY;
    _resolution.z = resolutionZ;
    return *this;
}

CellCenteredScalarGrid3::Builder&
CellCenteredScalarGrid3::Builder::withGridSpacing(const Vector3D& gridSpacing) {
    _gridSpacing = gridSpacing;
    return *this;
}

CellCenteredScalarGrid3::Builder&
CellCenteredScalarGrid3::Builder::withGridSpacing(
    double gridSpacingX, double gridSpacingY, double gridSpacingZ) {
    _gridSpacing.x = gridSpacingX;
    _gridSpacing.y = gridSpacingY;
    _gridSpacing.z = gridSpacingZ;
    return *this;
}

CellCenteredScalarGrid3::Builder&
CellCenteredScalarGrid3::Builder::withOrigin(const Vector3D& gridOrigin) {
    _gridOrigin = gridOrigin;
    return *this;
}

CellCenteredScalarGrid3::Builder&
CellCenteredScalarGrid3::Builder::withOrigin(
    double gridOriginX, double gridOriginY, double gridOriginZ) {
    _gridOrigin.x = gridOriginX;
    _gridOrigin.y = gridOriginY;
    _gridOrigin.z = gridOriginZ;
    return *this;
}

CellCenteredScalarGrid3::Builder&
CellCenteredScalarGrid3::Builder::withInitialValue(double initialVal) {
    _initialVal = initialVal;
    return *this;
}

CellCenteredScalarGrid3 CellCenteredScalarGrid3::Builder::build() const {
    return CellCenteredScalarGrid3(
        _resolution,
        _gridSpacing,
        _gridOrigin,
        _initialVal);
}

ScalarGrid3Ptr CellCenteredScalarGrid3::Builder::build(
    const Size3& resolution,
    const Vector3D& gridSpacing,
    const Vector3D& gridOrigin,
    double initialVal) const {
    return std::shared_ptr<CellCenteredScalarGrid3>(
        new CellCenteredScalarGrid3(
            resolution,
            gridSpacing,
            gridOrigin,
            initialVal),
        [] (CellCenteredScalarGrid3* obj) {
            delete obj;
        });
}

CellCenteredScalarGrid3Ptr
CellCenteredScalarGrid3::Builder::makeShared() const {
    return std::shared_ptr<CellCenteredScalarGrid3>(
        new CellCenteredScalarGrid3(
            _resolution,
            _gridSpacing,
            _gridOrigin,
            _initialVal),
        [] (CellCenteredScalarGrid3* obj) {
            delete obj;
        });
}
