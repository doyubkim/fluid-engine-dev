// Copyright (c) 2018 Doyub Kim
//
// I am making my contributions/submissions to this project solely in my
// personal capacity and am not conveying any rights to any intellectual
// property of any third parties.

#include <pch.h>

#include <jet/array_samplers2.h>
#include <jet/face_centered_grid2.h>
#include <jet/parallel.h>
#include <jet/serial.h>

#include <algorithm>
#include <utility>  // just make cpplint happy..
#include <vector>

using namespace jet;

FaceCenteredGrid2::FaceCenteredGrid2()
    : _dataOriginU(0.0, 0.5),
      _dataOriginV(0.5, 0.0),
      _uLinearSampler(LinearArraySampler2<double, double>(
          _dataU.constAccessor(), Vector2D(1, 1), _dataOriginU)),
      _vLinearSampler(LinearArraySampler2<double, double>(
          _dataV.constAccessor(), Vector2D(1, 1), _dataOriginV)) {}

FaceCenteredGrid2::FaceCenteredGrid2(size_t resolutionX, size_t resolutionY,
                                     double gridSpacingX, double gridSpacingY,
                                     double originX, double originY,
                                     double initialValueU, double initialValueV)
    : FaceCenteredGrid2(
          Size2(resolutionX, resolutionY), Vector2D(gridSpacingX, gridSpacingY),
          Vector2D(originX, originY), Vector2D(initialValueU, initialValueV)) {}

FaceCenteredGrid2::FaceCenteredGrid2(const Size2& resolution,
                                     const Vector2D& gridSpacing,
                                     const Vector2D& origin,
                                     const Vector2D& initialValue)
    : _uLinearSampler(LinearArraySampler2<double, double>(
          _dataU.constAccessor(), Vector2D(1, 1), _dataOriginU)),
      _vLinearSampler(LinearArraySampler2<double, double>(
          _dataV.constAccessor(), Vector2D(1, 1), _dataOriginV)) {
    resize(resolution, gridSpacing, origin, initialValue);
}

FaceCenteredGrid2::FaceCenteredGrid2(const FaceCenteredGrid2& other)
    : _uLinearSampler(LinearArraySampler2<double, double>(
          _dataU.constAccessor(), Vector2D(1, 1), _dataOriginU)),
      _vLinearSampler(LinearArraySampler2<double, double>(
          _dataV.constAccessor(), Vector2D(1, 1), _dataOriginV)) {
    set(other);
}

void FaceCenteredGrid2::swap(Grid2* other) {
    FaceCenteredGrid2* sameType = dynamic_cast<FaceCenteredGrid2*>(other);

    if (sameType != nullptr) {
        swapGrid(sameType);

        _dataU.swap(sameType->_dataU);
        _dataV.swap(sameType->_dataV);
        std::swap(_dataOriginU, sameType->_dataOriginU);
        std::swap(_dataOriginV, sameType->_dataOriginV);
        std::swap(_uLinearSampler, sameType->_uLinearSampler);
        std::swap(_vLinearSampler, sameType->_vLinearSampler);
        std::swap(_sampler, sameType->_sampler);
    }
}

void FaceCenteredGrid2::set(const FaceCenteredGrid2& other) {
    setGrid(other);

    _dataU.set(other._dataU);
    _dataV.set(other._dataV);
    _dataOriginU = other._dataOriginU;
    _dataOriginV = other._dataOriginV;

    resetSampler();
}

FaceCenteredGrid2& FaceCenteredGrid2::operator=(
    const FaceCenteredGrid2& other) {
    set(other);
    return *this;
}

double& FaceCenteredGrid2::u(size_t i, size_t j) { return _dataU(i, j); }

const double& FaceCenteredGrid2::u(size_t i, size_t j) const {
    return _dataU(i, j);
}

double& FaceCenteredGrid2::v(size_t i, size_t j) { return _dataV(i, j); }

const double& FaceCenteredGrid2::v(size_t i, size_t j) const {
    return _dataV(i, j);
}

Vector2D FaceCenteredGrid2::valueAtCellCenter(size_t i, size_t j) const {
    JET_ASSERT(i < resolution().x && j < resolution().y);

    return 0.5 * Vector2D(_dataU(i, j) + _dataU(i + 1, j),
                          _dataV(i, j) + _dataV(i, j + 1));
}

double FaceCenteredGrid2::divergenceAtCellCenter(size_t i, size_t j) const {
    JET_ASSERT(i < resolution().x && j < resolution().y);

    const Vector2D& gs = gridSpacing();

    double leftU = _dataU(i, j);
    double rightU = _dataU(i + 1, j);
    double bottomV = _dataV(i, j);
    double topV = _dataV(i, j + 1);

    return (rightU - leftU) / gs.x + (topV - bottomV) / gs.y;
}

double FaceCenteredGrid2::curlAtCellCenter(size_t i, size_t j) const {
    const Size2& res = resolution();

    JET_ASSERT(i < res.x && j < res.y);

    const Vector2D gs = gridSpacing();

    Vector2D left = valueAtCellCenter((i > 0) ? i - 1 : i, j);
    Vector2D right = valueAtCellCenter((i + 1 < res.x) ? i + 1 : i, j);
    Vector2D bottom = valueAtCellCenter(i, (j > 0) ? j - 1 : j);
    Vector2D top = valueAtCellCenter(i, (j + 1 < res.y) ? j + 1 : j);

    double Fx_ym = bottom.x;
    double Fx_yp = top.x;

    double Fy_xm = left.y;
    double Fy_xp = right.y;

    return 0.5 * (Fy_xp - Fy_xm) / gs.x - 0.5 * (Fx_yp - Fx_ym) / gs.y;
}

FaceCenteredGrid2::ScalarDataAccessor FaceCenteredGrid2::uAccessor() {
    return _dataU.accessor();
}

FaceCenteredGrid2::ConstScalarDataAccessor FaceCenteredGrid2::uConstAccessor()
    const {
    return _dataU.constAccessor();
}

FaceCenteredGrid2::ScalarDataAccessor FaceCenteredGrid2::vAccessor() {
    return _dataV.accessor();
}

FaceCenteredGrid2::ConstScalarDataAccessor FaceCenteredGrid2::vConstAccessor()
    const {
    return _dataV.constAccessor();
}

VectorGrid2::DataPositionFunc FaceCenteredGrid2::uPosition() const {
    Vector2D h = gridSpacing();

    return [this, h](size_t i, size_t j) -> Vector2D {
        return _dataOriginU + h * Vector2D({i, j});
    };
}

VectorGrid2::DataPositionFunc FaceCenteredGrid2::vPosition() const {
    Vector2D h = gridSpacing();

    return [this, h](size_t i, size_t j) -> Vector2D {
        return _dataOriginV + h * Vector2D({i, j});
    };
}

Size2 FaceCenteredGrid2::uSize() const { return _dataU.size(); }

Size2 FaceCenteredGrid2::vSize() const { return _dataV.size(); }

Vector2D FaceCenteredGrid2::uOrigin() const { return _dataOriginU; }

Vector2D FaceCenteredGrid2::vOrigin() const { return _dataOriginV; }

void FaceCenteredGrid2::fill(const Vector2D& value, ExecutionPolicy policy) {
    parallelFor(kZeroSize, _dataU.width(), kZeroSize, _dataU.height(),
                [this, value](size_t i, size_t j) { _dataU(i, j) = value.x; },
                policy);

    parallelFor(kZeroSize, _dataV.width(), kZeroSize, _dataV.height(),
                [this, value](size_t i, size_t j) { _dataV(i, j) = value.y; },
                policy);
}

void FaceCenteredGrid2::fill(
    const std::function<Vector2D(const Vector2D&)>& func,
    ExecutionPolicy policy) {
    DataPositionFunc uPos = uPosition();
    parallelFor(kZeroSize, _dataU.width(), kZeroSize, _dataU.height(),
                [this, &func, &uPos](size_t i, size_t j) {
                    _dataU(i, j) = func(uPos(i, j)).x;
                },
                policy);
    DataPositionFunc vPos = vPosition();
    parallelFor(kZeroSize, _dataV.width(), kZeroSize, _dataV.height(),
                [this, &func, &vPos](size_t i, size_t j) {
                    _dataV(i, j) = func(vPos(i, j)).y;
                },
                policy);
}

std::shared_ptr<VectorGrid2> FaceCenteredGrid2::clone() const {
    return CLONE_W_CUSTOM_DELETER(FaceCenteredGrid2);
}

void FaceCenteredGrid2::forEachUIndex(
    const std::function<void(size_t, size_t)>& func) const {
    _dataU.forEachIndex(func);
}

void FaceCenteredGrid2::parallelForEachUIndex(
    const std::function<void(size_t, size_t)>& func) const {
    _dataU.parallelForEachIndex(func);
}

void FaceCenteredGrid2::forEachVIndex(
    const std::function<void(size_t, size_t)>& func) const {
    _dataV.forEachIndex(func);
}

void FaceCenteredGrid2::parallelForEachVIndex(
    const std::function<void(size_t, size_t)>& func) const {
    _dataV.parallelForEachIndex(func);
}

Vector2D FaceCenteredGrid2::sample(const Vector2D& x) const {
    return _sampler(x);
}

std::function<Vector2D(const Vector2D&)> FaceCenteredGrid2::sampler() const {
    return _sampler;
}

double FaceCenteredGrid2::divergence(const Vector2D& x) const {
    ssize_t i, j;
    double fx, fy;
    Vector2D cellCenterOrigin = origin() + 0.5 * gridSpacing();

    Vector2D normalizedX = (x - cellCenterOrigin) / gridSpacing();

    getBarycentric(normalizedX.x, 0, static_cast<ssize_t>(resolution().x) - 1,
                   &i, &fx);
    getBarycentric(normalizedX.y, 0, static_cast<ssize_t>(resolution().y) - 1,
                   &j, &fy);

    std::array<Point2UI, 4> indices;
    std::array<double, 4> weights;

    indices[0] = Point2UI(i, j);
    indices[1] = Point2UI(i + 1, j);
    indices[2] = Point2UI(i, j + 1);
    indices[3] = Point2UI(i + 1, j + 1);

    weights[0] = (1.0 - fx) * (1.0 - fy);
    weights[1] = fx * (1.0 - fy);
    weights[2] = (1.0 - fx) * fy;
    weights[3] = fx * fy;

    double result = 0.0;

    for (int n = 0; n < 4; ++n) {
        result +=
            weights[n] * divergenceAtCellCenter(indices[n].x, indices[n].y);
    }

    return result;
}

double FaceCenteredGrid2::curl(const Vector2D& x) const {
    ssize_t i, j;
    double fx, fy;
    Vector2D cellCenterOrigin = origin() + 0.5 * gridSpacing();

    Vector2D normalizedX = (x - cellCenterOrigin) / gridSpacing();

    getBarycentric(normalizedX.x, 0, static_cast<ssize_t>(resolution().x) - 1,
                   &i, &fx);
    getBarycentric(normalizedX.y, 0, static_cast<ssize_t>(resolution().y) - 1,
                   &j, &fy);

    std::array<Point2UI, 4> indices;
    std::array<double, 4> weights;

    indices[0] = Point2UI(i, j);
    indices[1] = Point2UI(i + 1, j);
    indices[2] = Point2UI(i, j + 1);
    indices[3] = Point2UI(i + 1, j + 1);

    weights[0] = (1.0 - fx) * (1.0 - fy);
    weights[1] = fx * (1.0 - fy);
    weights[2] = (1.0 - fx) * fy;
    weights[3] = fx * fy;

    double result = 0.0;

    for (int n = 0; n < 4; ++n) {
        result += weights[n] * curlAtCellCenter(indices[n].x, indices[n].y);
    }

    return result;
}

void FaceCenteredGrid2::onResize(const Size2& resolution,
                                 const Vector2D& gridSpacing,
                                 const Vector2D& origin,
                                 const Vector2D& initialValue) {
    if (resolution != Size2(0, 0)) {
        _dataU.resize(resolution + Size2(1, 0), initialValue.x);
        _dataV.resize(resolution + Size2(0, 1), initialValue.y);
    } else {
        _dataU.resize(Size2(0, 0));
        _dataV.resize(Size2(0, 0));
    }
    _dataOriginU = origin + 0.5 * Vector2D(0.0, gridSpacing.y);
    _dataOriginV = origin + 0.5 * Vector2D(gridSpacing.x, 0.0);

    resetSampler();
}

void FaceCenteredGrid2::resetSampler() {
    LinearArraySampler2<double, double> uSampler(_dataU.constAccessor(),
                                                 gridSpacing(), _dataOriginU);
    LinearArraySampler2<double, double> vSampler(_dataV.constAccessor(),
                                                 gridSpacing(), _dataOriginV);

    _uLinearSampler = uSampler;
    _vLinearSampler = vSampler;

    _sampler = [uSampler, vSampler](const Vector2D& x) -> Vector2D {
        double u = uSampler(x);
        double v = vSampler(x);
        return Vector2D(u, v);
    };
}

FaceCenteredGrid2::Builder FaceCenteredGrid2::builder() { return Builder(); }

void FaceCenteredGrid2::getData(std::vector<double>* data) const {
    size_t size = uSize().x * uSize().y + vSize().x * vSize().y;
    data->resize(size);
    size_t cnt = 0;
    _dataU.forEach([&](double value) { (*data)[cnt++] = value; });
    _dataV.forEach([&](double value) { (*data)[cnt++] = value; });
}

void FaceCenteredGrid2::setData(const std::vector<double>& data) {
    JET_ASSERT(uSize().x * uSize().y + vSize().x * vSize().y == data.size());

    size_t cnt = 0;
    _dataU.forEachIndex(
        [&](size_t i, size_t j) { _dataU(i, j) = data[cnt++]; });
    _dataV.forEachIndex(
        [&](size_t i, size_t j) { _dataV(i, j) = data[cnt++]; });
}

FaceCenteredGrid2::Builder& FaceCenteredGrid2::Builder::withResolution(
    const Size2& resolution) {
    _resolution = resolution;
    return *this;
}

FaceCenteredGrid2::Builder& FaceCenteredGrid2::Builder::withResolution(
    size_t resolutionX, size_t resolutionY) {
    _resolution.x = resolutionX;
    _resolution.y = resolutionY;
    return *this;
}

FaceCenteredGrid2::Builder& FaceCenteredGrid2::Builder::withGridSpacing(
    const Vector2D& gridSpacing) {
    _gridSpacing = gridSpacing;
    return *this;
}

FaceCenteredGrid2::Builder& FaceCenteredGrid2::Builder::withGridSpacing(
    double gridSpacingX, double gridSpacingY) {
    _gridSpacing.x = gridSpacingX;
    _gridSpacing.y = gridSpacingY;
    return *this;
}

FaceCenteredGrid2::Builder& FaceCenteredGrid2::Builder::withOrigin(
    const Vector2D& gridOrigin) {
    _gridOrigin = gridOrigin;
    return *this;
}

FaceCenteredGrid2::Builder& FaceCenteredGrid2::Builder::withOrigin(
    double gridOriginX, double gridOriginY) {
    _gridOrigin.x = gridOriginX;
    _gridOrigin.y = gridOriginY;
    return *this;
}

FaceCenteredGrid2::Builder& FaceCenteredGrid2::Builder::withInitialValue(
    const Vector2D& initialVal) {
    _initialVal = initialVal;
    return *this;
}

FaceCenteredGrid2::Builder& FaceCenteredGrid2::Builder::withInitialValue(
    double initialValX, double initialValY) {
    _initialVal.x = initialValX;
    _initialVal.y = initialValY;
    return *this;
}

FaceCenteredGrid2 FaceCenteredGrid2::Builder::build() const {
    return FaceCenteredGrid2(_resolution, _gridSpacing, _gridOrigin,
                             _initialVal);
}

FaceCenteredGrid2Ptr FaceCenteredGrid2::Builder::makeShared() const {
    return std::shared_ptr<FaceCenteredGrid2>(
        new FaceCenteredGrid2(_resolution, _gridSpacing, _gridOrigin,
                              _initialVal),
        [](FaceCenteredGrid2* obj) { delete obj; });
}

VectorGrid2Ptr FaceCenteredGrid2::Builder::build(
    const Size2& resolution, const Vector2D& gridSpacing,
    const Vector2D& gridOrigin, const Vector2D& initialVal) const {
    return std::shared_ptr<FaceCenteredGrid2>(
        new FaceCenteredGrid2(resolution, gridSpacing, gridOrigin, initialVal),
        [](FaceCenteredGrid2* obj) { delete obj; });
}
