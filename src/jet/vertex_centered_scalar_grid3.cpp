// Copyright (c) 2016 Doyub Kim

#include <pch.h>
#include <jet/vertex_centered_scalar_grid3.h>
#include <utility>  // just make cpplint happy..

using namespace jet;

VertexCenteredScalarGrid3::VertexCenteredScalarGrid3() {
}

VertexCenteredScalarGrid3::VertexCenteredScalarGrid3(
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

VertexCenteredScalarGrid3::VertexCenteredScalarGrid3(
    const Size3& resolution,
    const Vector3D& gridSpacing,
    const Vector3D& origin,
    double initialValue) {
    resize(resolution, gridSpacing, origin, initialValue);
}

VertexCenteredScalarGrid3::VertexCenteredScalarGrid3(
    const VertexCenteredScalarGrid3& other) {
    set(other);
}

Size3 VertexCenteredScalarGrid3::dataSize() const {
    if (resolution() != Size3(0, 0, 0)) {
        return resolution() + Size3(1, 1, 1);
    } else {
        return Size3(0, 0, 0);
    }
}

Vector3D VertexCenteredScalarGrid3::dataOrigin() const {
    return origin();
}

std::shared_ptr<ScalarGrid3> VertexCenteredScalarGrid3::clone() const {
    return std::make_shared<VertexCenteredScalarGrid3>(*this);
}

void VertexCenteredScalarGrid3::swap(Grid3* other) {
    VertexCenteredScalarGrid3* sameType
        = dynamic_cast<VertexCenteredScalarGrid3*>(other);
    if (sameType != nullptr) {
        swapScalarGrid(sameType);
    }
}

void VertexCenteredScalarGrid3::set(const VertexCenteredScalarGrid3& other) {
    setScalarGrid(other);
}

VertexCenteredScalarGrid3&
VertexCenteredScalarGrid3::operator=(const VertexCenteredScalarGrid3& other) {
    set(other);
    return *this;
}

ScalarGridBuilder3Ptr VertexCenteredScalarGrid3::builder() {
    return std::make_shared<VertexCenteredScalarGridBuilder3>();
}
