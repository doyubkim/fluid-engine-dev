// Copyright (c) 2016 Doyub Kim

#include <pch.h>
#include <jet/fdm_utils.h>
#include <jet/parallel.h>
#include <jet/scalar_grid2.h>
#include <jet/serial.h>

#include <algorithm>

using namespace jet;

ScalarGrid2::ScalarGrid2() :
    _linearSampler(
        LinearArraySampler2<double, double>(
            _data.constAccessor(),
            Vector2D(1, 1),
            Vector2D())) {
}

ScalarGrid2::~ScalarGrid2() {
}

void ScalarGrid2::clear() {
    resize(Size2(), gridSpacing(), origin(), 0.0);
}

void ScalarGrid2::resize(
    size_t resolutionX,
    size_t resolutionY,
    double gridSpacingX,
    double gridSpacingY,
    double originX,
    double originY,
    double initialValue) {
    resize(
        Size2(resolutionX, resolutionY),
        Vector2D(gridSpacingX, gridSpacingY),
        Vector2D(originX, originY),
        initialValue);
}

void ScalarGrid2::resize(
    const Size2& resolution,
    const Vector2D& gridSpacing,
    const Vector2D& origin,
    double initialValue) {
    setSizeParameters(resolution, gridSpacing, origin);

    _data.resize(dataSize(), initialValue);
    resetSampler();
}

void ScalarGrid2::resize(
    double gridSpacingX,
    double gridSpacingY,
    double originX,
    double originY) {
    resize(
        Vector2D(gridSpacingX, gridSpacingY),
        Vector2D(originX, originY));
}

void ScalarGrid2::resize(
    const Vector2D& gridSpacing,
    const Vector2D& origin) {
    resize(resolution(), gridSpacing, origin);
}

const double& ScalarGrid2::operator()(size_t i, size_t j) const {
    return _data(i, j);
}

double& ScalarGrid2::operator()(size_t i, size_t j) {
    return _data(i, j);
}

Vector2D ScalarGrid2::gradientAtDataPoint(size_t i, size_t j) const {
    return gradient2(_data.constAccessor(), gridSpacing(), i, j);
}

double ScalarGrid2::laplacianAtDataPoint(size_t i, size_t j) const {
    return laplacian2(_data.constAccessor(), gridSpacing(), i, j);
}

double ScalarGrid2::sample(const Vector2D& x) const {
    return _sampler(x);
}

std::function<double(const Vector2D&)> ScalarGrid2::sampler() const {
    return _sampler;
}

Vector2D ScalarGrid2::gradient(const Vector2D& x) const {
    std::array<Point2UI, 4> indices;
    std::array<double, 4> weights;
    _linearSampler.getCoordinatesAndWeights(x, &indices, &weights);

    Vector2D result;

    for (int i = 0; i < 4; ++i) {
        result += weights[i] * gradientAtDataPoint(indices[i].x, indices[i].y);
    }

    return result;
}

double ScalarGrid2::laplacian(const Vector2D& x) const {
    std::array<Point2UI, 4> indices;
    std::array<double, 4> weights;
    _linearSampler.getCoordinatesAndWeights(x, &indices, &weights);

    double result = 0.0;

    for (int i = 0; i < 4; ++i) {
        result += weights[i]
            * laplacianAtDataPoint(indices[i].x, indices[i].y);
    }

    return result;
}

ScalarGrid2::ScalarDataAccessor ScalarGrid2::dataAccessor() {
    return _data.accessor();
}

ScalarGrid2::ConstScalarDataAccessor ScalarGrid2::constDataAccessor() const {
    return _data.constAccessor();
}

ScalarGrid2::DataPositionFunc ScalarGrid2::dataPosition() const {
    Vector2D o = dataOrigin();
    return [this, o](size_t i, size_t j) -> Vector2D {
        return o + gridSpacing() * Vector2D({i, j});
    };
}

void ScalarGrid2::fill(double value) {
    parallelFor(
        kZeroSize, _data.width(),
        kZeroSize, _data.height(),
        [this, value](size_t i, size_t j) {
            _data(i, j) = value;
        });
}

void ScalarGrid2::fill(const std::function<double(const Vector2D&)>& func) {
    DataPositionFunc pos = dataPosition();
    parallelFor(
        kZeroSize, _data.width(),
        kZeroSize, _data.height(),
        [this, &func, &pos](size_t i, size_t j) {
            _data(i, j) = func(pos(i, j));
        });
}

void ScalarGrid2::forEachDataPointIndex(
    const std::function<void(size_t, size_t)>& func) const {
    _data.forEachIndex(func);
}

void ScalarGrid2::parallelForEachDataPointIndex(
    const std::function<void(size_t, size_t)>& func) const {
    _data.parallelForEachIndex(func);
}

void ScalarGrid2::serialize(std::ostream* strm) const {
    serializeGrid(strm);
    _data.serialize(strm);
}

void ScalarGrid2::deserialize(std::istream* strm) {
    deserializeGrid(strm);
    _data.deserialize(strm);

    resetSampler();
}

void ScalarGrid2::swapScalarGrid(ScalarGrid2* other) {
    swapGrid(other);

    _data.swap(other->_data);
    std::swap(_linearSampler, other->_linearSampler);
    std::swap(_sampler, other->_sampler);
}

void ScalarGrid2::setScalarGrid(const ScalarGrid2& other) {
    setGrid(other);

    _data.set(other._data);
    resetSampler();
}

void ScalarGrid2::resetSampler() {
    _linearSampler = LinearArraySampler2<double, double>(
        _data.constAccessor(), gridSpacing(), dataOrigin());
    _sampler = _linearSampler.functor();
}

ScalarGridBuilder2::ScalarGridBuilder2() {
}

ScalarGridBuilder2::~ScalarGridBuilder2() {
}
