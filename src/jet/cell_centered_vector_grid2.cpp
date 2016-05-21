// Copyright (c) 2016 Doyub Kim

#include <pch.h>
#include <jet/cell_centered_vector_grid2.h>
#include <jet/parallel.h>
#include <algorithm>  // just make cpplint happy..

using namespace jet;

CellCenteredVectorGrid2::CellCenteredVectorGrid2() {
}

CellCenteredVectorGrid2::CellCenteredVectorGrid2(
    size_t resolutionX,
    size_t resolutionY,
    double gridSpacingX,
    double gridSpacingY,
    double originX,
    double originY,
    double initialValueU,
    double initialValueV) {
    resize(
        resolutionX,
        resolutionY,
        gridSpacingX,
        gridSpacingY,
        originX,
        originY,
        initialValueU,
        initialValueV);
}

CellCenteredVectorGrid2::CellCenteredVectorGrid2(
    const Size2& resolution,
    const Vector2D& gridSpacing,
    const Vector2D& origin,
    const Vector2D& initialValue) {
    resize(resolution, gridSpacing, origin, initialValue);
}

CellCenteredVectorGrid2::CellCenteredVectorGrid2(
    const CellCenteredVectorGrid2& other) {
    set(other);
}

Size2 CellCenteredVectorGrid2::dataSize() const {
    return resolution();
}

Vector2D CellCenteredVectorGrid2::dataOrigin() const {
    return origin() + 0.5 * gridSpacing();
}

void CellCenteredVectorGrid2::swap(Grid2* other) {
    CellCenteredVectorGrid2* sameType
        = dynamic_cast<CellCenteredVectorGrid2*>(other);
    if (sameType != nullptr) {
        swapCollocatedVectorGrid(sameType);
    }
}

void CellCenteredVectorGrid2::set(const CellCenteredVectorGrid2& other) {
    setCollocatedVectorGrid(other);
}

CellCenteredVectorGrid2&
CellCenteredVectorGrid2::operator=(const CellCenteredVectorGrid2& other) {
    set(other);
    return *this;
}

void CellCenteredVectorGrid2::fill(const Vector2D& value) {
    Size2 size = dataSize();
    auto acc = dataAccessor();
    parallelFor(
        kZeroSize, size.x,
        kZeroSize, size.y,
        [this, value, &acc](size_t i, size_t j) {
            acc(i, j) = value;
        });
}

void CellCenteredVectorGrid2::fill(const std::function<Vector2D(
    const Vector2D&)>& func) {
    Size2 size = dataSize();
    auto acc = dataAccessor();
    DataPositionFunc pos = dataPosition();
    parallelFor(
        kZeroSize, size.x,
        kZeroSize, size.y,
        [this, &func, &acc, &pos](size_t i, size_t j) {
            acc(i, j) = func(pos(i, j));
        });
}

std::shared_ptr<VectorGrid2> CellCenteredVectorGrid2::clone() const {
    return std::make_shared<CellCenteredVectorGrid2>(*this);
}

VectorGridBuilder2Ptr CellCenteredVectorGrid2::builder() {
    return std::make_shared<CellCenteredVectorGridBuilder2>();
}


CellCenteredVectorGridBuilder2::CellCenteredVectorGridBuilder2() {
}

VectorGrid2Ptr CellCenteredVectorGridBuilder2::build(
    const Size2& resolution,
    const Vector2D& gridSpacing,
    const Vector2D& gridOrigin,
    const Vector2D& initialVal) const {
    return std::make_shared<CellCenteredVectorGrid2>(
        resolution,
        gridSpacing,
        gridOrigin,
        initialVal);
}
