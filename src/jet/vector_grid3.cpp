// Copyright (c) 2016 Doyub Kim

#include <pch.h>
#include <jet/array_samplers3.h>
#include <jet/vector_grid3.h>

using namespace jet;

VectorGrid3::VectorGrid3() {
}

VectorGrid3::~VectorGrid3() {
}

void VectorGrid3::clear() {
    resize(Size3(), gridSpacing(), origin(), Vector3D());
}

void VectorGrid3::resize(
    size_t resolutionX,
    size_t resolutionY,
    size_t resolutionZ,
    double gridSpacingX,
    double gridSpacingY,
    double gridSpacingZ,
    double originX,
    double originY,
    double originZ,
    double initialValueX,
    double initialValueY,
    double initialValueZ) {
    resize(
        Size3(resolutionX, resolutionY, resolutionZ),
        Vector3D(gridSpacingX, gridSpacingY, gridSpacingZ),
        Vector3D(originX, originY, originZ),
        Vector3D(initialValueX, initialValueY, initialValueZ));
}

void VectorGrid3::resize(
    const Size3& resolution,
    const Vector3D& gridSpacing,
    const Vector3D& origin,
    const Vector3D& initialValue) {
    setSizeParameters(resolution, gridSpacing, origin);

    onResize(resolution, gridSpacing, origin, initialValue);
}

void VectorGrid3::resize(
    double gridSpacingX,
    double gridSpacingY,
    double gridSpacingZ,
    double originX,
    double originY,
    double originZ) {
    resize(
        Vector3D(gridSpacingX, gridSpacingY, gridSpacingZ),
        Vector3D(originX, originY, originZ));
}

void VectorGrid3::resize(const Vector3D& gridSpacing, const Vector3D& origin) {
    resize(resolution(), gridSpacing, origin);
}


VectorGridBuilder3::VectorGridBuilder3() {
}

VectorGridBuilder3::~VectorGridBuilder3() {
}
