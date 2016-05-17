// Copyright (c) 2016 Doyub Kim

#include <pch.h>
#include <jet/fdm_utils.h>
#include <jet/parallel.h>
#include <jet/scalar_grid3.h>
#include <jet/serial.h>

#include <algorithm>

using namespace jet;

ScalarGrid3::ScalarGrid3() :
    _linearSampler(
        LinearArraySampler3<double, double>(
            _data.constAccessor(),
            Vector3D(1, 1, 1),
            Vector3D())) {
}

ScalarGrid3::~ScalarGrid3() {
}

void ScalarGrid3::clear() {
    resize(Size3(), gridSpacing(), origin(), 0.0);
}

void ScalarGrid3::resize(
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
        Size3(resolutionX, resolutionY, resolutionZ),
        Vector3D(gridSpacingX, gridSpacingY, gridSpacingZ),
        Vector3D(originX, originY, originZ),
        initialValue);
}

void ScalarGrid3::resize(
    const Size3& resolution,
    const Vector3D& gridSpacing,
    const Vector3D& origin,
    double initialValue) {
    setSizeParameters(resolution, gridSpacing, origin);

    _data.resize(dataSize(), initialValue);
    resetSampler();
}


void ScalarGrid3::resize(
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

void ScalarGrid3::resize(
    const Vector3D& gridSpacing,
    const Vector3D& origin) {
    resize(resolution(), gridSpacing, origin);
}

const double& ScalarGrid3::operator()(size_t i, size_t j, size_t k) const {
    return _data(i, j, k);
}

double& ScalarGrid3::operator()(size_t i, size_t j, size_t k) {
    return _data(i, j, k);
}

Vector3D ScalarGrid3::gradientAtDataPoint(size_t i, size_t j, size_t k) const {
    return gradient3(_data.constAccessor(), gridSpacing(), i, j, k);
}

double ScalarGrid3::laplacianAtDataPoint(size_t i, size_t j, size_t k) const {
    return laplacian3(_data.constAccessor(), gridSpacing(), i, j, k);
}

double ScalarGrid3::sample(const Vector3D& x) const {
    return _sampler(x);
}

std::function<double(const Vector3D&)> ScalarGrid3::sampler() const {
    return _sampler;
}

Vector3D ScalarGrid3::gradient(const Vector3D& x) const {
    std::array<Point3UI, 8> indices;
    std::array<double, 8> weights;
    _linearSampler.getCoordinatesAndWeights(x, &indices, &weights);

    Vector3D result;

    for (int i = 0; i < 8; ++i) {
        result += weights[i] * gradientAtDataPoint(
            indices[i].x, indices[i].y, indices[i].z);
    }

    return result;
}

double ScalarGrid3::laplacian(const Vector3D& x) const {
    std::array<Point3UI, 8> indices;
    std::array<double, 8> weights;
    _linearSampler.getCoordinatesAndWeights(x, &indices, &weights);

    double result = 0.0;

    for (int i = 0; i < 8; ++i) {
        result += weights[i]
            * laplacianAtDataPoint(indices[i].x, indices[i].y, indices[i].z);
    }

    return result;
}

ScalarGrid3::ScalarDataAccessor ScalarGrid3::dataAccessor() {
    return _data.accessor();
}

ScalarGrid3::ConstScalarDataAccessor ScalarGrid3::constDataAccessor() const {
    return _data.constAccessor();
}

ScalarGrid3::DataPositionFunc ScalarGrid3::dataPosition() const {
    Vector3D o = dataOrigin();
    return [this, o](size_t i, size_t j, size_t k) -> Vector3D {
        return o + gridSpacing() * Vector3D({i, j, k});
    };
}

void ScalarGrid3::fill(double value) {
    parallelFor(
        kZeroSize, _data.width(),
        kZeroSize, _data.height(),
        kZeroSize, _data.depth(),
        [this, value](size_t i, size_t j, size_t k) {
            _data(i, j, k) = value;
        });
}

void ScalarGrid3::fill(const std::function<double(const Vector3D&)>& func) {
    DataPositionFunc pos = dataPosition();
    parallelFor(
        kZeroSize, _data.width(),
        kZeroSize, _data.height(),
        kZeroSize, _data.depth(),
        [this, &func, &pos](size_t i, size_t j, size_t k) {
            _data(i, j, k) = func(pos(i, j, k));
        });
}

void ScalarGrid3::forEachDataPointIndex(
    const std::function<void(size_t, size_t, size_t)>& func) const {
    _data.forEachIndex(func);
}

void ScalarGrid3::parallelForEachDataPointIndex(
    const std::function<void(size_t, size_t, size_t)>& func) const {
    _data.parallelForEachIndex(func);
}

void ScalarGrid3::serialize(std::ostream* strm) const {
    serializeGrid(strm);
    _data.serialize(strm);
}

void ScalarGrid3::deserialize(std::istream* strm) {
    deserializeGrid(strm);
    _data.deserialize(strm);

    resetSampler();
}

void ScalarGrid3::swapScalarGrid(ScalarGrid3* other) {
    swapGrid(other);

    _data.swap(other->_data);
    std::swap(_linearSampler, other->_linearSampler);
    std::swap(_sampler, other->_sampler);
}

void ScalarGrid3::setScalarGrid(const ScalarGrid3& other) {
    setGrid(other);

    _data.set(other._data);
    resetSampler();
}

void ScalarGrid3::resetSampler() {
    _linearSampler = LinearArraySampler3<double, double>(
        _data.constAccessor(), gridSpacing(), dataOrigin());
    _sampler = _linearSampler.functor();
}

ScalarGridBuilder3::ScalarGridBuilder3() {
}

ScalarGridBuilder3::~ScalarGridBuilder3() {
}
