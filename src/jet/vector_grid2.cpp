// Copyright (c) 2016 Doyub Kim

#include <pch.h>
#include <jet/array_samplers2.h>
#include <jet/vector_grid2.h>

using namespace jet;

VectorGrid2::VectorGrid2() {
}

VectorGrid2::~VectorGrid2() {
}

void VectorGrid2::clear() {
    resize(Size2(), gridSpacing(), origin(), Vector2D());
}

void VectorGrid2::resize(
    size_t resolutionX,
    size_t resolutionY,
    double gridSpacingX,
    double gridSpacingY,
    double originX,
    double originY,
    double initialValueX,
    double initialValueY) {
    resize(
        Size2(resolutionX, resolutionY),
        Vector2D(gridSpacingX, gridSpacingY),
        Vector2D(originX, originY),
        Vector2D(initialValueX, initialValueY));
}

void VectorGrid2::resize(
    const Size2& resolution,
    const Vector2D& gridSpacing,
    const Vector2D& origin,
    const Vector2D& initialValue) {
    setSizeParameters(resolution, gridSpacing, origin);

    onResize(resolution, gridSpacing, origin, initialValue);
}

void VectorGrid2::resize(
    double gridSpacingX,
    double gridSpacingY,
    double originX,
    double originY) {
    resize(
        Vector2D(gridSpacingX, gridSpacingY),
        Vector2D(originX, originY));
}

void VectorGrid2::resize(const Vector2D& gridSpacing, const Vector2D& origin) {
    resize(resolution(), gridSpacing, origin);
}


VectorGridBuilder2::VectorGridBuilder2() {
}

VectorGridBuilder2::~VectorGridBuilder2() {
}
