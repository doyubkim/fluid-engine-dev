// Copyright (c) 2018 Doyub Kim
//
// I am making my contributions/submissions to this project solely in my
// personal capacity and am not conveying any rights to any intellectual
// property of any third parties.

#include <pch.h>

#include <jet/cell_centered_vector_grid3.h>
#include <jet/parallel.h>

#include <utility>  // just make cpplint happy..

using namespace jet;

CellCenteredVectorGrid3::CellCenteredVectorGrid3() {}

CellCenteredVectorGrid3::CellCenteredVectorGrid3(
    size_t resolutionX, size_t resolutionY, size_t resolutionZ,
    double gridSpacingX, double gridSpacingY, double gridSpacingZ,
    double originX, double originY, double originZ, double initialValueU,
    double initialValueV, double initialValueW) {
    resize(resolutionX, resolutionY, resolutionZ, gridSpacingX, gridSpacingY,
           gridSpacingZ, originX, originY, originZ, initialValueU,
           initialValueV, initialValueW);
}

CellCenteredVectorGrid3::CellCenteredVectorGrid3(const Size3& resolution,
                                                 const Vector3D& gridSpacing,
                                                 const Vector3D& origin,
                                                 const Vector3D& initialValue) {
    resize(resolution, gridSpacing, origin, initialValue);
}

CellCenteredVectorGrid3::CellCenteredVectorGrid3(
    const CellCenteredVectorGrid3& other) {
    set(other);
}

Size3 CellCenteredVectorGrid3::dataSize() const { return resolution(); }

Vector3D CellCenteredVectorGrid3::dataOrigin() const {
    return origin() + 0.5 * gridSpacing();
}

void CellCenteredVectorGrid3::swap(Grid3* other) {
    CellCenteredVectorGrid3* sameType =
        dynamic_cast<CellCenteredVectorGrid3*>(other);
    if (sameType != nullptr) {
        swapCollocatedVectorGrid(sameType);
    }
}

void CellCenteredVectorGrid3::set(const CellCenteredVectorGrid3& other) {
    setCollocatedVectorGrid(other);
}

CellCenteredVectorGrid3& CellCenteredVectorGrid3::operator=(
    const CellCenteredVectorGrid3& other) {
    set(other);
    return *this;
}

void CellCenteredVectorGrid3::fill(const Vector3D& value,
                                   ExecutionPolicy policy) {
    Size3 size = dataSize();
    auto acc = dataAccessor();
    parallelFor(
        kZeroSize, size.x, kZeroSize, size.y, kZeroSize, size.z,
        [value, &acc](size_t i, size_t j, size_t k) { acc(i, j, k) = value; },
        policy);
}

void CellCenteredVectorGrid3::fill(
    const std::function<Vector3D(const Vector3D&)>& func,
    ExecutionPolicy policy) {
    Size3 size = dataSize();
    auto acc = dataAccessor();
    DataPositionFunc pos = dataPosition();
    parallelFor(kZeroSize, size.x, kZeroSize, size.y, kZeroSize, size.z,
                [&func, &acc, &pos](size_t i, size_t j, size_t k) {
                    acc(i, j, k) = func(pos(i, j, k));
                },
                policy);
}

std::shared_ptr<VectorGrid3> CellCenteredVectorGrid3::clone() const {
    return CLONE_W_CUSTOM_DELETER(CellCenteredVectorGrid3);
}

CellCenteredVectorGrid3::Builder CellCenteredVectorGrid3::builder() {
    return Builder();
}

CellCenteredVectorGrid3::Builder&
CellCenteredVectorGrid3::Builder::withResolution(const Size3& resolution) {
    _resolution = resolution;
    return *this;
}

CellCenteredVectorGrid3::Builder&
CellCenteredVectorGrid3::Builder::withResolution(size_t resolutionX,
                                                 size_t resolutionY,
                                                 size_t resolutionZ) {
    _resolution.x = resolutionX;
    _resolution.y = resolutionY;
    _resolution.z = resolutionZ;
    return *this;
}

CellCenteredVectorGrid3::Builder&
CellCenteredVectorGrid3::Builder::withGridSpacing(const Vector3D& gridSpacing) {
    _gridSpacing = gridSpacing;
    return *this;
}

CellCenteredVectorGrid3::Builder&
CellCenteredVectorGrid3::Builder::withGridSpacing(double gridSpacingX,
                                                  double gridSpacingY,
                                                  double gridSpacingZ) {
    _gridSpacing.x = gridSpacingX;
    _gridSpacing.y = gridSpacingY;
    _gridSpacing.z = gridSpacingZ;
    return *this;
}

CellCenteredVectorGrid3::Builder& CellCenteredVectorGrid3::Builder::withOrigin(
    const Vector3D& gridOrigin) {
    _gridOrigin = gridOrigin;
    return *this;
}

CellCenteredVectorGrid3::Builder& CellCenteredVectorGrid3::Builder::withOrigin(
    double gridOriginX, double gridOriginY, double gridOriginZ) {
    _gridOrigin.x = gridOriginX;
    _gridOrigin.y = gridOriginY;
    _gridOrigin.z = gridOriginZ;
    return *this;
}

CellCenteredVectorGrid3::Builder&
CellCenteredVectorGrid3::Builder::withInitialValue(const Vector3D& initialVal) {
    _initialVal = initialVal;
    return *this;
}

CellCenteredVectorGrid3::Builder&
CellCenteredVectorGrid3::Builder::withInitialValue(double initialValX,
                                                   double initialValY,
                                                   double initialValZ) {
    _initialVal.x = initialValX;
    _initialVal.y = initialValY;
    _initialVal.z = initialValZ;
    return *this;
}

CellCenteredVectorGrid3 CellCenteredVectorGrid3::Builder::build() const {
    return CellCenteredVectorGrid3(_resolution, _gridSpacing, _gridOrigin,
                                   _initialVal);
}

VectorGrid3Ptr CellCenteredVectorGrid3::Builder::build(
    const Size3& resolution, const Vector3D& gridSpacing,
    const Vector3D& gridOrigin, const Vector3D& initialVal) const {
    return std::shared_ptr<CellCenteredVectorGrid3>(
        new CellCenteredVectorGrid3(resolution, gridSpacing, gridOrigin,
                                    initialVal),
        [](CellCenteredVectorGrid3* obj) { delete obj; });
}

CellCenteredVectorGrid3Ptr CellCenteredVectorGrid3::Builder::makeShared()
    const {
    return std::shared_ptr<CellCenteredVectorGrid3>(
        new CellCenteredVectorGrid3(_resolution, _gridSpacing, _gridOrigin,
                                    _initialVal),
        [](CellCenteredVectorGrid3* obj) { delete obj; });
}
