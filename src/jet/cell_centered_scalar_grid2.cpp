// Copyright (c) 2016 Doyub Kim

#include <pch.h>
#include <jet/cell_centered_scalar_grid2.h>

#include <algorithm>

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
    return std::make_shared<CellCenteredScalarGrid2>(*this);
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

ScalarGridBuilder2Ptr CellCenteredScalarGrid2::builder() {
    return std::make_shared<CellCenteredScalarGridBuilder2>();
}


CellCenteredScalarGridBuilder2::CellCenteredScalarGridBuilder2() {
}

ScalarGrid2Ptr CellCenteredScalarGridBuilder2::build(
        const Size2& resolution,
        const Vector2D& gridSpacing,
        const Vector2D& gridOrigin,
        double initialVal) const {
    return std::make_shared<CellCenteredScalarGrid2>(
        resolution,
        gridSpacing,
        gridOrigin,
        initialVal);
}
