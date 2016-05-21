// Copyright (c) 2016 Doyub Kim

#include <pch.h>
#include <jet/vertex_centered_scalar_grid2.h>
#include <algorithm>  // just make cpplint happy..

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

VertexCenteredScalarGrid2::~VertexCenteredScalarGrid2() {
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
    return std::make_shared<VertexCenteredScalarGrid2>(*this);
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


ScalarGridBuilder2Ptr VertexCenteredScalarGrid2::builder() {
    return std::make_shared<VertexCenteredScalarGridBuilder2>();
}


VertexCenteredScalarGridBuilder2::VertexCenteredScalarGridBuilder2() {
}

ScalarGrid2Ptr VertexCenteredScalarGridBuilder2::build(
        const Size2& resolution,
        const Vector2D& gridSpacing,
        const Vector2D& gridOrigin,
        double initialVal) const {
    return std::make_shared<VertexCenteredScalarGrid2>(
        resolution,
        gridSpacing,
        gridOrigin,
        initialVal);
}
