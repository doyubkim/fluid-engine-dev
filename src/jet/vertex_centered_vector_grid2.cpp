// Copyright (c) 2018 Doyub Kim
//
// I am making my contributions/submissions to this project solely in my
// personal capacity and am not conveying any rights to any intellectual
// property of any third parties.

#include <jet/array_samplers.h>
#include <jet/parallel.h>
#include <jet/vertex_centered_vector_grid2.h>
#include <pch.h>
#include <utility>  // just make cpplint happy..

using namespace jet;

VertexCenteredVectorGrid2::VertexCenteredVectorGrid2() {}

VertexCenteredVectorGrid2::VertexCenteredVectorGrid2(
    size_t resolutionX, size_t resolutionY, double gridSpacingX,
    double gridSpacingY, double originX, double originY, double initialValueU,
    double initialValueV) {
    resize(resolutionX, resolutionY, gridSpacingX, gridSpacingY, originX,
           originY, initialValueU, initialValueV);
}

VertexCenteredVectorGrid2::VertexCenteredVectorGrid2(
    const Vector2UZ& resolution, const Vector2D& gridSpacing,
    const Vector2D& origin, const Vector2D& initialValue) {
    resize(resolution, gridSpacing, origin, initialValue);
}

VertexCenteredVectorGrid2::VertexCenteredVectorGrid2(
    const VertexCenteredVectorGrid2& other) {
    set(other);
}

Vector2UZ VertexCenteredVectorGrid2::dataSize() const {
    if (resolution() != Vector2UZ(0, 0)) {
        return resolution() + Vector2UZ(1, 1);
    } else {
        return Vector2UZ(0, 0);
    }
}

Vector2D VertexCenteredVectorGrid2::dataOrigin() const { return origin(); }

void VertexCenteredVectorGrid2::swap(Grid2* other) {
    VertexCenteredVectorGrid2* sameType =
        dynamic_cast<VertexCenteredVectorGrid2*>(other);
    if (sameType != nullptr) {
        swapCollocatedVectorGrid(sameType);
    }
}

void VertexCenteredVectorGrid2::set(const VertexCenteredVectorGrid2& other) {
    setCollocatedVectorGrid(other);
}

VertexCenteredVectorGrid2& VertexCenteredVectorGrid2::operator=(
    const VertexCenteredVectorGrid2& other) {
    set(other);
    return *this;
}

void VertexCenteredVectorGrid2::fill(const Vector2D& value,
                                     ExecutionPolicy policy) {
    Vector2UZ size = dataSize();
    auto acc = dataView();
    parallelFor(kZeroSize, size.x, kZeroSize, size.y,
                [value, &acc](size_t i, size_t j) { acc(i, j) = value; },
                policy);
}

void VertexCenteredVectorGrid2::fill(
    const std::function<Vector2D(const Vector2D&)>& func,
    ExecutionPolicy policy) {
    Vector2UZ size = dataSize();
    auto acc = dataView();
    DataPositionFunc pos = dataPosition();
    parallelFor(kZeroSize, size.x, kZeroSize, size.y,
                [&func, &acc, &pos](size_t i, size_t j) {
                    acc(i, j) = func(pos(i, j));
                },
                policy);
}

std::shared_ptr<VectorGrid2> VertexCenteredVectorGrid2::clone() const {
    return CLONE_W_CUSTOM_DELETER(VertexCenteredVectorGrid2);
}

VertexCenteredVectorGrid2::Builder VertexCenteredVectorGrid2::builder() {
    return Builder();
}

VertexCenteredVectorGrid2::Builder&
VertexCenteredVectorGrid2::Builder::withResolution(const Vector2UZ& resolution) {
    _resolution = resolution;
    return *this;
}

VertexCenteredVectorGrid2::Builder&
VertexCenteredVectorGrid2::Builder::withResolution(size_t resolutionX,
                                                   size_t resolutionY) {
    _resolution.x = resolutionX;
    _resolution.y = resolutionY;
    return *this;
}

VertexCenteredVectorGrid2::Builder&
VertexCenteredVectorGrid2::Builder::withGridSpacing(
    const Vector2D& gridSpacing) {
    _gridSpacing = gridSpacing;
    return *this;
}

VertexCenteredVectorGrid2::Builder&
VertexCenteredVectorGrid2::Builder::withGridSpacing(double gridSpacingX,
                                                    double gridSpacingY) {
    _gridSpacing.x = gridSpacingX;
    _gridSpacing.y = gridSpacingY;
    return *this;
}

VertexCenteredVectorGrid2::Builder&
VertexCenteredVectorGrid2::Builder::withOrigin(const Vector2D& gridOrigin) {
    _gridOrigin = gridOrigin;
    return *this;
}

VertexCenteredVectorGrid2::Builder&
VertexCenteredVectorGrid2::Builder::withOrigin(double gridOriginX,
                                               double gridOriginY) {
    _gridOrigin.x = gridOriginX;
    _gridOrigin.y = gridOriginY;
    return *this;
}

VertexCenteredVectorGrid2::Builder&
VertexCenteredVectorGrid2::Builder::withInitialValue(
    const Vector2D& initialVal) {
    _initialVal = initialVal;
    return *this;
}

VertexCenteredVectorGrid2::Builder&
VertexCenteredVectorGrid2::Builder::withInitialValue(double initialValX,
                                                     double initialValY) {
    _initialVal.x = initialValX;
    _initialVal.y = initialValY;
    return *this;
}

VertexCenteredVectorGrid2 VertexCenteredVectorGrid2::Builder::build() const {
    return VertexCenteredVectorGrid2(_resolution, _gridSpacing, _gridOrigin,
                                     _initialVal);
}

VertexCenteredVectorGrid2Ptr VertexCenteredVectorGrid2::Builder::makeShared()
    const {
    return std::shared_ptr<VertexCenteredVectorGrid2>(
        new VertexCenteredVectorGrid2(_resolution, _gridSpacing, _gridOrigin,
                                      _initialVal),
        [](VertexCenteredVectorGrid2* obj) { delete obj; });
}

VectorGrid2Ptr VertexCenteredVectorGrid2::Builder::build(
    const Vector2UZ& resolution, const Vector2D& gridSpacing,
    const Vector2D& gridOrigin, const Vector2D& initialVal) const {
    return std::shared_ptr<VertexCenteredVectorGrid2>(
        new VertexCenteredVectorGrid2(resolution, gridSpacing, gridOrigin,
                                      initialVal),
        [](VertexCenteredVectorGrid2* obj) { delete obj; });
}
