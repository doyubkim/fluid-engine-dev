// Copyright (c) 2016 Doyub Kim

#include <pch.h>
#include <jet/array_samplers3.h>
#include <jet/parallel.h>
#include <jet/vertex_centered_vector_grid3.h>
#include <utility>  // just make cpplint happy..

using namespace jet;

VertexCenteredVectorGrid3::VertexCenteredVectorGrid3() {
}

VertexCenteredVectorGrid3::VertexCenteredVectorGrid3(
    size_t resolutionX,
    size_t resolutionY,
    size_t resolutionZ,
    double gridSpacingX,
    double gridSpacingY,
    double gridSpacingZ,
    double originX,
    double originY,
    double originZ,
    double initialValueU,
    double initialValueV,
    double initialValueW) {
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
        initialValueU,
        initialValueV,
        initialValueW);
}

VertexCenteredVectorGrid3::VertexCenteredVectorGrid3(
    const Size3& resolution,
    const Vector3D& gridSpacing,
    const Vector3D& origin,
    const Vector3D& initialValue) {
    resize(resolution, gridSpacing, origin, initialValue);
}

Size3 VertexCenteredVectorGrid3::dataSize() const {
    if (resolution() != Size3(0, 0, 0)) {
        return resolution() + Size3(1, 1, 1);
    } else {
        return Size3(0, 0, 0);
    }
}

Vector3D VertexCenteredVectorGrid3::dataOrigin() const {
    return origin();
}

void VertexCenteredVectorGrid3::swap(Grid3* other) {
    VertexCenteredVectorGrid3* sameType
        = dynamic_cast<VertexCenteredVectorGrid3*>(other);
    if (sameType != nullptr) {
        swapCollocatedVectorGrid(sameType);
    }
}

void VertexCenteredVectorGrid3::fill(const Vector3D& value) {
    Size3 size = dataSize();
    auto acc = dataAccessor();
    parallelFor(
        kZeroSize, size.x,
        kZeroSize, size.y,
        kZeroSize, size.z,
        [this, value, &acc](size_t i, size_t j, size_t k) {
            acc(i, j, k) = value;
        });
}

void VertexCenteredVectorGrid3::fill(
    const std::function<Vector3D(const Vector3D&)>& func) {
    Size3 size = dataSize();
    auto acc = dataAccessor();
    DataPositionFunc pos = dataPosition();
    parallelFor(
        kZeroSize, size.x,
        kZeroSize, size.y,
        kZeroSize, size.z,
        [this, &func, &acc, &pos](size_t i, size_t j, size_t k) {
            acc(i, j, k) = func(pos(i, j, k));
        });
}

std::shared_ptr<VectorGrid3> VertexCenteredVectorGrid3::clone() const {
    return std::make_shared<VertexCenteredVectorGrid3>(*this);
}

void VertexCenteredVectorGrid3::set(const VertexCenteredVectorGrid3& other) {
    setCollocatedVectorGrid(other);
}

VertexCenteredVectorGrid3& VertexCenteredVectorGrid3::operator=(
    const VertexCenteredVectorGrid3& other) {
    set(other);
    return *this;
}

VertexCenteredVectorGrid3::Builder VertexCenteredVectorGrid3::builder() {
    return Builder();
}


VertexCenteredVectorGrid3::Builder&
VertexCenteredVectorGrid3::Builder::withResolution(const Size3& resolution) {
    _resolution = resolution;
    return *this;
}

VertexCenteredVectorGrid3::Builder&
VertexCenteredVectorGrid3::Builder::withResolution(
    size_t resolutionX, size_t resolutionY, size_t resolutionZ) {
    _resolution.x = resolutionX;
    _resolution.y = resolutionY;
    _resolution.z = resolutionZ;
    return *this;
}

VertexCenteredVectorGrid3::Builder&
VertexCenteredVectorGrid3::Builder::withGridSpacing(
    const Vector3D& gridSpacing) {
    _gridSpacing = gridSpacing;
    return *this;
}

VertexCenteredVectorGrid3::Builder&
VertexCenteredVectorGrid3::Builder::withGridSpacing(
    double gridSpacingX, double gridSpacingY, double gridSpacingZ) {
    _gridSpacing.x = gridSpacingX;
    _gridSpacing.y = gridSpacingY;
    _gridSpacing.z = gridSpacingZ;
    return *this;
}

VertexCenteredVectorGrid3::Builder&
VertexCenteredVectorGrid3::Builder::withGridOrigin(const Vector3D& gridOrigin) {
    _gridOrigin = gridOrigin;
    return *this;
}

VertexCenteredVectorGrid3::Builder&
VertexCenteredVectorGrid3::Builder::withGridOrigin(
    double gridOriginX, double gridOriginY, double gridOriginZ) {
    _gridOrigin.x = gridOriginX;
    _gridOrigin.y = gridOriginY;
    _gridOrigin.z = gridOriginZ;
    return *this;
}

VertexCenteredVectorGrid3::Builder&
VertexCenteredVectorGrid3::Builder::withInitialValue(
    const Vector3D& initialVal) {
    _initialVal = initialVal;
    return *this;
}

VertexCenteredVectorGrid3::Builder&
VertexCenteredVectorGrid3::Builder::withInitialValue(
    double initialValX, double initialValY, double initialValZ) {
    _initialVal.x = initialValX;
    _initialVal.y = initialValY;
    _initialVal.z = initialValZ;
    return *this;
}

VertexCenteredVectorGrid3 VertexCenteredVectorGrid3::Builder::build() const {
    return VertexCenteredVectorGrid3(
        _resolution,
        _gridSpacing,
        _gridOrigin,
        _initialVal);
}

VertexCenteredVectorGrid3Ptr
VertexCenteredVectorGrid3::Builder::makeShared() const {
    return std::shared_ptr<VertexCenteredVectorGrid3>(
        new VertexCenteredVectorGrid3(
            _resolution,
            _gridSpacing,
            _gridOrigin,
            _initialVal),
        [] (VertexCenteredVectorGrid3* obj) {
            delete obj;
        });
}

VectorGrid3Ptr VertexCenteredVectorGrid3::Builder::build(
    const Size3& resolution,
    const Vector3D& gridSpacing,
    const Vector3D& gridOrigin,
    const Vector3D& initialVal) const {
    return std::shared_ptr<VertexCenteredVectorGrid3>(
        new VertexCenteredVectorGrid3(
            resolution,
            gridSpacing,
            gridOrigin,
            initialVal),
        [] (VertexCenteredVectorGrid3* obj) {
            delete obj;
        });
}
