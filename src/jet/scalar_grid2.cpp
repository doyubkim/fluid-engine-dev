// Copyright (c) 2018 Doyub Kim
//
// I am making my contributions/submissions to this project solely in my
// personal capacity and am not conveying any rights to any intellectual
// property of any third parties.

#ifdef _MSC_VER
#pragma warning(disable: 4244)
#endif

#include <pch.h>

#include <fbs_helpers.h>
#include <generated/scalar_grid2_generated.h>

#include <jet/fdm_utils.h>
#include <jet/parallel.h>
#include <jet/scalar_grid2.h>
#include <jet/serial.h>

#include <flatbuffers/flatbuffers.h>

#include <algorithm>
#include <string>
#include <utility>  // just make cpplint happy..
#include <vector>

using namespace jet;

ScalarGrid2::ScalarGrid2()
    : _linearSampler(LinearArraySampler2<double, double>(
          _data.constAccessor(), Vector2D(1, 1), Vector2D())) {}

ScalarGrid2::~ScalarGrid2() {}

void ScalarGrid2::clear() { resize(Size2(), gridSpacing(), origin(), 0.0); }

void ScalarGrid2::resize(size_t resolutionX, size_t resolutionY,
                         double gridSpacingX, double gridSpacingY,
                         double originX, double originY, double initialValue) {
    resize(Size2(resolutionX, resolutionY),
           Vector2D(gridSpacingX, gridSpacingY), Vector2D(originX, originY),
           initialValue);
}

void ScalarGrid2::resize(const Size2& resolution, const Vector2D& gridSpacing,
                         const Vector2D& origin, double initialValue) {
    setSizeParameters(resolution, gridSpacing, origin);

    _data.resize(dataSize(), initialValue);
    resetSampler();
}

void ScalarGrid2::resize(double gridSpacingX, double gridSpacingY,
                         double originX, double originY) {
    resize(Vector2D(gridSpacingX, gridSpacingY), Vector2D(originX, originY));
}

void ScalarGrid2::resize(const Vector2D& gridSpacing, const Vector2D& origin) {
    resize(resolution(), gridSpacing, origin);
}

const double& ScalarGrid2::operator()(size_t i, size_t j) const {
    return _data(i, j);
}

double& ScalarGrid2::operator()(size_t i, size_t j) { return _data(i, j); }

Vector2D ScalarGrid2::gradientAtDataPoint(size_t i, size_t j) const {
    return gradient2(_data.constAccessor(), gridSpacing(), i, j);
}

double ScalarGrid2::laplacianAtDataPoint(size_t i, size_t j) const {
    return laplacian2(_data.constAccessor(), gridSpacing(), i, j);
}

double ScalarGrid2::sample(const Vector2D& x) const { return _sampler(x); }

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
        result += weights[i] * laplacianAtDataPoint(indices[i].x, indices[i].y);
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

void ScalarGrid2::fill(double value, ExecutionPolicy policy) {
    parallelFor(kZeroSize, _data.width(), kZeroSize, _data.height(),
                [this, value](size_t i, size_t j) { _data(i, j) = value; },
                policy);
}

void ScalarGrid2::fill(const std::function<double(const Vector2D&)>& func,
                       ExecutionPolicy policy) {
    DataPositionFunc pos = dataPosition();
    parallelFor(kZeroSize, _data.width(), kZeroSize, _data.height(),
                [this, &func, &pos](size_t i, size_t j) {
                    _data(i, j) = func(pos(i, j));
                },
                policy);
}

void ScalarGrid2::forEachDataPointIndex(
    const std::function<void(size_t, size_t)>& func) const {
    _data.forEachIndex(func);
}

void ScalarGrid2::parallelForEachDataPointIndex(
    const std::function<void(size_t, size_t)>& func) const {
    _data.parallelForEachIndex(func);
}

void ScalarGrid2::serialize(std::vector<uint8_t>* buffer) const {
    flatbuffers::FlatBufferBuilder builder(1024);

    auto fbsResolution = jetToFbs(resolution());
    auto fbsGridSpacing = jetToFbs(gridSpacing());
    auto fbsOrigin = jetToFbs(origin());

    std::vector<double> gridData;
    getData(&gridData);
    auto data = builder.CreateVector(gridData.data(), gridData.size());

    auto fbsGrid = fbs::CreateScalarGrid2(builder, &fbsResolution,
                                          &fbsGridSpacing, &fbsOrigin, data);

    builder.Finish(fbsGrid);

    uint8_t* buf = builder.GetBufferPointer();
    size_t size = builder.GetSize();

    buffer->resize(size);
    memcpy(buffer->data(), buf, size);
}

void ScalarGrid2::deserialize(const std::vector<uint8_t>& buffer) {
    auto fbsGrid = fbs::GetScalarGrid2(buffer.data());

    resize(fbsToJet(*fbsGrid->resolution()), fbsToJet(*fbsGrid->gridSpacing()),
           fbsToJet(*fbsGrid->origin()));

    auto data = fbsGrid->data();
    std::vector<double> gridData(data->size());
    std::copy(data->begin(), data->end(), gridData.begin());

    setData(gridData);
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

void ScalarGrid2::getData(std::vector<double>* data) const {
    size_t size = dataSize().x * dataSize().y;
    data->resize(size);
    std::copy(_data.begin(), _data.end(), data->begin());
}

void ScalarGrid2::setData(const std::vector<double>& data) {
    JET_ASSERT(dataSize().x * dataSize().y == data.size());

    std::copy(data.begin(), data.end(), _data.begin());
}

ScalarGridBuilder2::ScalarGridBuilder2() {}

ScalarGridBuilder2::~ScalarGridBuilder2() {}
