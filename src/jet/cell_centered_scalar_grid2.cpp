// Copyright (c) 2018 Doyub Kim
//
// I am making my contributions/submissions to this project solely in my
// personal capacity and am not conveying any rights to any intellectual
// property of any third parties.

#include <pch.h>
#include <jet/cell_centered_scalar_grid2.h>

#include <algorithm>
#include <utility>  // just make cpplint happy..

using namespace jet;

CellCenteredScalarGrid2::CellCenteredScalarGrid2() {
}

CellCenteredScalarGrid2::CellCenteredScalarGrid2(
    size_t resolutionX,
    size_t resolutionY,
    double gridSpacingX,
    double gridSpacingY,
    double originX,
    double originY,
    double initialValue) {
    resize(
        resolutionX,
        resolutionY,
        gridSpacingX,
        gridSpacingY,
        originX,
        originY,
        initialValue);
}

CellCenteredScalarGrid2::CellCenteredScalarGrid2(
    const Size2& resolution,
    const Vector2D& gridSpacing,
    const Vector2D& origin,
    double initialValue) {
    resize(resolution, gridSpacing, origin, initialValue);
}

CellCenteredScalarGrid2::CellCenteredScalarGrid2(
    const CellCenteredScalarGrid2& other) {
    set(other);
}

Size2 CellCenteredScalarGrid2::dataSize() const {
    // The size of the data should be the same as the grid resolution.
    return resolution();
}

Vector2D CellCenteredScalarGrid2::dataOrigin() const {
    return origin() + 0.5 * gridSpacing();
}

std::shared_ptr<ScalarGrid2> CellCenteredScalarGrid2::clone() const {
    return CLONE_W_CUSTOM_DELETER(CellCenteredScalarGrid2);
}

void CellCenteredScalarGrid2::swap(Grid2* other) {
    CellCenteredScalarGrid2* sameType
        = dynamic_cast<CellCenteredScalarGrid2*>(other);
    if (sameType != nullptr) {
        swapScalarGrid(sameType);
    }
}

void CellCenteredScalarGrid2::set(const CellCenteredScalarGrid2& other) {
    setScalarGrid(other);
}

CellCenteredScalarGrid2&
CellCenteredScalarGrid2::operator=(const CellCenteredScalarGrid2& other) {
    set(other);
    return *this;
}

CellCenteredScalarGrid2::Builder CellCenteredScalarGrid2::builder() {
    return Builder();
}


CellCenteredScalarGrid2::Builder&
CellCenteredScalarGrid2::Builder::withResolution(const Size2& resolution) {
    _resolution = resolution;
    return *this;
}

CellCenteredScalarGrid2::Builder&
CellCenteredScalarGrid2::Builder::withResolution(
    size_t resolutionX, size_t resolutionY) {
    _resolution.x = resolutionX;
    _resolution.y = resolutionY;
    return *this;
}

CellCenteredScalarGrid2::Builder&
CellCenteredScalarGrid2::Builder::withGridSpacing(const Vector2D& gridSpacing) {
    _gridSpacing = gridSpacing;
    return *this;
}

CellCenteredScalarGrid2::Builder&
CellCenteredScalarGrid2::Builder::withGridSpacing(
    double gridSpacingX, double gridSpacingY) {
    _gridSpacing.x = gridSpacingX;
    _gridSpacing.y = gridSpacingY;
    return *this;
}

CellCenteredScalarGrid2::Builder&
CellCenteredScalarGrid2::Builder::withOrigin(const Vector2D& gridOrigin) {
    _gridOrigin = gridOrigin;
    return *this;
}

CellCenteredScalarGrid2::Builder&
CellCenteredScalarGrid2::Builder::withOrigin(
    double gridOriginX, double gridOriginY) {
    _gridOrigin.x = gridOriginX;
    _gridOrigin.y = gridOriginY;
    return *this;
}

CellCenteredScalarGrid2::Builder&
CellCenteredScalarGrid2::Builder::withInitialValue(double initialVal) {
    _initialVal = initialVal;
    return *this;
}

CellCenteredScalarGrid2 CellCenteredScalarGrid2::Builder::build() const {
    return CellCenteredScalarGrid2(
        _resolution,
        _gridSpacing,
        _gridOrigin,
        _initialVal);
}

ScalarGrid2Ptr CellCenteredScalarGrid2::Builder::build(
    const Size2& resolution,
    const Vector2D& gridSpacing,
    const Vector2D& gridOrigin,
    double initialVal) const {
    return std::shared_ptr<CellCenteredScalarGrid2>(
        new CellCenteredScalarGrid2(
            resolution,
            gridSpacing,
            gridOrigin,
            initialVal),
        [] (CellCenteredScalarGrid2* obj) {
            delete obj;
        });
}

CellCenteredScalarGrid2Ptr
CellCenteredScalarGrid2::Builder::makeShared() const {
    return std::shared_ptr<CellCenteredScalarGrid2>(
        new CellCenteredScalarGrid2(
            _resolution,
            _gridSpacing,
            _gridOrigin,
            _initialVal),
        [] (CellCenteredScalarGrid2* obj) {
            delete obj;
        });
}
