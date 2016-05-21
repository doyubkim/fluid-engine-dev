// Copyright (c) 2016 Doyub Kim

#include <pch.h>
#include <jet/array_samplers2.h>
#include <jet/parallel.h>
#include <jet/vertex_centered_vector_grid2.h>
#include <algorithm>  // just make cpplint happy..

using namespace jet;

VertexCenteredVectorGrid2::VertexCenteredVectorGrid2() {
}

VertexCenteredVectorGrid2::VertexCenteredVectorGrid2(
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

VertexCenteredVectorGrid2::VertexCenteredVectorGrid2(
    const Size2& resolution,
    const Vector2D& gridSpacing,
    const Vector2D& origin,
    const Vector2D& initialValue) {
    resize(resolution, gridSpacing, origin, initialValue);
}

VertexCenteredVectorGrid2::VertexCenteredVectorGrid2(
    const VertexCenteredVectorGrid2& other) {
    set(other);
}

VertexCenteredVectorGrid2::~VertexCenteredVectorGrid2() {
}

Size2 VertexCenteredVectorGrid2::dataSize() const {
    if (resolution() != Size2(0, 0)) {
        return resolution() + Size2(1, 1);
    } else {
        return Size2(0, 0);
    }
}

Vector2D VertexCenteredVectorGrid2::dataOrigin() const {
    return origin();
}

void VertexCenteredVectorGrid2::swap(Grid2* other) {
    VertexCenteredVectorGrid2* sameType
        = dynamic_cast<VertexCenteredVectorGrid2*>(other);
    if (sameType != nullptr) {
        swapCollocatedVectorGrid(sameType);
    }
}

void VertexCenteredVectorGrid2::set(const VertexCenteredVectorGrid2& other) {
    setCollocatedVectorGrid(other);
}

VertexCenteredVectorGrid2&
VertexCenteredVectorGrid2::operator=(const VertexCenteredVectorGrid2& other) {
    set(other);
    return *this;
}

void VertexCenteredVectorGrid2::fill(const Vector2D& value) {
    Size2 size = dataSize();
    auto acc = dataAccessor();
    parallelFor(
        kZeroSize, size.x,
        kZeroSize, size.y,
        [this, value, &acc](size_t i, size_t j) {
            acc(i, j) = value;
        });
}

void VertexCenteredVectorGrid2::fill(
    const std::function<Vector2D(const Vector2D&)>& func) {
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

std::shared_ptr<VectorGrid2> VertexCenteredVectorGrid2::clone() const {
    return std::make_shared<VertexCenteredVectorGrid2>(*this);
}

VectorGridBuilder2Ptr VertexCenteredVectorGrid2::builder() {
    return std::make_shared<VertexCenteredVectorGridBuilder2>();
}


VertexCenteredVectorGridBuilder2::VertexCenteredVectorGridBuilder2() {
}

VectorGrid2Ptr VertexCenteredVectorGridBuilder2::build(
    const Size2& resolution,
    const Vector2D& gridSpacing,
    const Vector2D& gridOrigin,
    const Vector2D& initialVal) const {
    return std::make_shared<VertexCenteredVectorGrid2>(
        resolution,
        gridSpacing,
        gridOrigin,
        initialVal);
}
