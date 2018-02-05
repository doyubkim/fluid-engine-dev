// Copyright (c) 2018 Doyub Kim
//
// I am making my contributions/submissions to this project solely in my
// personal capacity and am not conveying any rights to any intellectual
// property of any third parties.

#include <pch.h>
#include <jet/vertex_centered_scalar_grid2.h>
#include <utility>  // just make cpplint happy..

using namespace jet;

VertexCenteredScalarGrid2::VertexCenteredScalarGrid2() {
}

VertexCenteredScalarGrid2::VertexCenteredScalarGrid2(
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

VertexCenteredScalarGrid2::VertexCenteredScalarGrid2(
    const Size2& resolution,
    const Vector2D& gridSpacing,
    const Vector2D& origin,
    double initialValue) {
    resize(resolution, gridSpacing, origin, initialValue);
}

VertexCenteredScalarGrid2::VertexCenteredScalarGrid2(
    const VertexCenteredScalarGrid2& other) {
    set(other);
}

Size2 VertexCenteredScalarGrid2::dataSize() const {
    if (resolution() != Size2(0, 0)) {
        return resolution() + Size2(1, 1);
    } else {
        return Size2(0, 0);
    }
}

Vector2D VertexCenteredScalarGrid2::dataOrigin() const {
    return origin();
}

std::shared_ptr<ScalarGrid2> VertexCenteredScalarGrid2::clone() const {
    return CLONE_W_CUSTOM_DELETER(VertexCenteredScalarGrid2);
}

void VertexCenteredScalarGrid2::swap(Grid2* other) {
    VertexCenteredScalarGrid2* sameType
        = dynamic_cast<VertexCenteredScalarGrid2*>(other);
    if (sameType != nullptr) {
        swapScalarGrid(sameType);
    }
}

void VertexCenteredScalarGrid2::set(const VertexCenteredScalarGrid2& other) {
    setScalarGrid(other);
}

VertexCenteredScalarGrid2&
VertexCenteredScalarGrid2::operator=(const VertexCenteredScalarGrid2& other) {
    set(other);
    return *this;
}

VertexCenteredScalarGrid2::Builder VertexCenteredScalarGrid2::builder() {
    return Builder();
}


VertexCenteredScalarGrid2::Builder&
VertexCenteredScalarGrid2::Builder::withResolution(const Size2& resolution) {
    _resolution = resolution;
    return *this;
}

VertexCenteredScalarGrid2::Builder&
VertexCenteredScalarGrid2::Builder::withResolution(
    size_t resolutionX, size_t resolutionY) {
    _resolution.x = resolutionX;
    _resolution.y = resolutionY;
    return *this;
}

VertexCenteredScalarGrid2::Builder&
VertexCenteredScalarGrid2::Builder::withGridSpacing(
    const Vector2D& gridSpacing) {
    _gridSpacing = gridSpacing;
    return *this;
}

VertexCenteredScalarGrid2::Builder&
VertexCenteredScalarGrid2::Builder::withGridSpacing(
    double gridSpacingX, double gridSpacingY) {
    _gridSpacing.x = gridSpacingX;
    _gridSpacing.y = gridSpacingY;
    return *this;
}

VertexCenteredScalarGrid2::Builder&
VertexCenteredScalarGrid2::Builder::withOrigin(const Vector2D& gridOrigin) {
    _gridOrigin = gridOrigin;
    return *this;
}

VertexCenteredScalarGrid2::Builder&
VertexCenteredScalarGrid2::Builder::withOrigin(
    double gridOriginX, double gridOriginY) {
    _gridOrigin.x = gridOriginX;
    _gridOrigin.y = gridOriginY;
    return *this;
}

VertexCenteredScalarGrid2::Builder&
VertexCenteredScalarGrid2::Builder::withInitialValue(double initialVal) {
    _initialVal = initialVal;
    return *this;
}

VertexCenteredScalarGrid2 VertexCenteredScalarGrid2::Builder::build() const {
    return VertexCenteredScalarGrid2(
        _resolution,
        _gridSpacing,
        _gridOrigin,
        _initialVal);
}

VertexCenteredScalarGrid2Ptr
VertexCenteredScalarGrid2::Builder::makeShared() const {
    return std::shared_ptr<VertexCenteredScalarGrid2>(
        new VertexCenteredScalarGrid2(
            _resolution,
            _gridSpacing,
            _gridOrigin,
            _initialVal),
        [] (VertexCenteredScalarGrid2* obj) {
            delete obj;
        });
}

ScalarGrid2Ptr VertexCenteredScalarGrid2::Builder::build(
    const Size2& resolution,
    const Vector2D& gridSpacing,
    const Vector2D& gridOrigin,
    double initialVal) const {
    return std::shared_ptr<VertexCenteredScalarGrid2>(
        new VertexCenteredScalarGrid2(
            resolution,
            gridSpacing,
            gridOrigin,
            initialVal),
        [] (VertexCenteredScalarGrid2* obj) {
            delete obj;
        });
}
