// Copyright (c) 2018 Doyub Kim
//
// I am making my contributions/submissions to this project solely in my
// personal capacity and am not conveying any rights to any intellectual
// property of any third parties.

#include <pch.h>

#include <jet/cell_centered_vector_grid2.h>
#include <jet/parallel.h>

#include <utility>  // just make cpplint happy..

using namespace jet;

CellCenteredVectorGrid2::CellCenteredVectorGrid2() {}

CellCenteredVectorGrid2::CellCenteredVectorGrid2(
    size_t resolutionX, size_t resolutionY, double gridSpacingX,
    double gridSpacingY, double originX, double originY, double initialValueU,
    double initialValueV) {
    resize(resolutionX, resolutionY, gridSpacingX, gridSpacingY, originX,
           originY, initialValueU, initialValueV);
}

CellCenteredVectorGrid2::CellCenteredVectorGrid2(const Size2& resolution,
                                                 const Vector2D& gridSpacing,
                                                 const Vector2D& origin,
                                                 const Vector2D& initialValue) {
    resize(resolution, gridSpacing, origin, initialValue);
}

CellCenteredVectorGrid2::CellCenteredVectorGrid2(
    const CellCenteredVectorGrid2& other) {
    set(other);
}

Size2 CellCenteredVectorGrid2::dataSize() const { return resolution(); }

Vector2D CellCenteredVectorGrid2::dataOrigin() const {
    return origin() + 0.5 * gridSpacing();
}

void CellCenteredVectorGrid2::swap(Grid2* other) {
    CellCenteredVectorGrid2* sameType =
        dynamic_cast<CellCenteredVectorGrid2*>(other);
    if (sameType != nullptr) {
        swapCollocatedVectorGrid(sameType);
    }
}

void CellCenteredVectorGrid2::set(const CellCenteredVectorGrid2& other) {
    setCollocatedVectorGrid(other);
}

CellCenteredVectorGrid2& CellCenteredVectorGrid2::operator=(
    const CellCenteredVectorGrid2& other) {
    set(other);
    return *this;
}

void CellCenteredVectorGrid2::fill(const Vector2D& value,
                                   ExecutionPolicy policy) {
    Size2 size = dataSize();
    auto acc = dataAccessor();
    parallelFor(kZeroSize, size.x, kZeroSize, size.y,
                [value, &acc](size_t i, size_t j) { acc(i, j) = value; },
                policy);
}

void CellCenteredVectorGrid2::fill(
    const std::function<Vector2D(const Vector2D&)>& func,
    ExecutionPolicy policy) {
    Size2 size = dataSize();
    auto acc = dataAccessor();
    DataPositionFunc pos = dataPosition();
    parallelFor(kZeroSize, size.x, kZeroSize, size.y,
                [&func, &acc, &pos](size_t i, size_t j) {
                    acc(i, j) = func(pos(i, j));
                },
                policy);
}

std::shared_ptr<VectorGrid2> CellCenteredVectorGrid2::clone() const {
    return CLONE_W_CUSTOM_DELETER(CellCenteredVectorGrid2);
}

CellCenteredVectorGrid2::Builder CellCenteredVectorGrid2::builder() {
    return Builder();
}

CellCenteredVectorGrid2::Builder&
CellCenteredVectorGrid2::Builder::withResolution(const Size2& resolution) {
    _resolution = resolution;
    return *this;
}

CellCenteredVectorGrid2::Builder&
CellCenteredVectorGrid2::Builder::withResolution(size_t resolutionX,
                                                 size_t resolutionY) {
    _resolution.x = resolutionX;
    _resolution.y = resolutionY;
    return *this;
}

CellCenteredVectorGrid2::Builder&
CellCenteredVectorGrid2::Builder::withGridSpacing(const Vector2D& gridSpacing) {
    _gridSpacing = gridSpacing;
    return *this;
}

CellCenteredVectorGrid2::Builder&
CellCenteredVectorGrid2::Builder::withGridSpacing(double gridSpacingX,
                                                  double gridSpacingY) {
    _gridSpacing.x = gridSpacingX;
    _gridSpacing.y = gridSpacingY;
    return *this;
}

CellCenteredVectorGrid2::Builder& CellCenteredVectorGrid2::Builder::withOrigin(
    const Vector2D& gridOrigin) {
    _gridOrigin = gridOrigin;
    return *this;
}

CellCenteredVectorGrid2::Builder& CellCenteredVectorGrid2::Builder::withOrigin(
    double gridOriginX, double gridOriginY) {
    _gridOrigin.x = gridOriginX;
    _gridOrigin.y = gridOriginY;
    return *this;
}

CellCenteredVectorGrid2::Builder&
CellCenteredVectorGrid2::Builder::withInitialValue(const Vector2D& initialVal) {
    _initialVal = initialVal;
    return *this;
}

CellCenteredVectorGrid2::Builder&
CellCenteredVectorGrid2::Builder::withInitialValue(double initialValX,
                                                   double initialValY) {
    _initialVal.x = initialValX;
    _initialVal.y = initialValY;
    return *this;
}

CellCenteredVectorGrid2 CellCenteredVectorGrid2::Builder::build() const {
    return CellCenteredVectorGrid2(_resolution, _gridSpacing, _gridOrigin,
                                   _initialVal);
}

VectorGrid2Ptr CellCenteredVectorGrid2::Builder::build(
    const Size2& resolution, const Vector2D& gridSpacing,
    const Vector2D& gridOrigin, const Vector2D& initialVal) const {
    return std::shared_ptr<CellCenteredVectorGrid2>(
        new CellCenteredVectorGrid2(resolution, gridSpacing, gridOrigin,
                                    initialVal),
        [](CellCenteredVectorGrid2* obj) { delete obj; });
}

CellCenteredVectorGrid2Ptr CellCenteredVectorGrid2::Builder::makeShared()
    const {
    return std::shared_ptr<CellCenteredVectorGrid2>(
        new CellCenteredVectorGrid2(_resolution, _gridSpacing, _gridOrigin,
                                    _initialVal),
        [](CellCenteredVectorGrid2* obj) { delete obj; });
}
