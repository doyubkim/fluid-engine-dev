// Copyright (c) 2018 Doyub Kim
//
// I am making my contributions/submissions to this project solely in my
// personal capacity and am not conveying any rights to any intellectual
// property of any third parties.

#include <pch.h>

#include <jet/array_samplers3.h>
#include <jet/face_centered_grid3.h>
#include <jet/parallel.h>
#include <jet/serial.h>

#include <algorithm>
#include <utility>  // just make cpplint happy..
#include <vector>

using namespace jet;

FaceCenteredGrid3::FaceCenteredGrid3()
    : _dataOriginU(0.0, 0.5, 0.5),
      _dataOriginV(0.5, 0.0, 0.5),
      _dataOriginW(0.5, 0.5, 0.0),
      _uLinearSampler(LinearArraySampler3<double, double>(
          _dataU.constAccessor(), Vector3D(1, 1, 1), _dataOriginU)),
      _vLinearSampler(LinearArraySampler3<double, double>(
          _dataV.constAccessor(), Vector3D(1, 1, 1), _dataOriginV)),
      _wLinearSampler(LinearArraySampler3<double, double>(
          _dataW.constAccessor(), Vector3D(1, 1, 1), _dataOriginW)) {}

FaceCenteredGrid3::FaceCenteredGrid3(size_t resolutionX, size_t resolutionY,
                                     size_t resolutionZ, double gridSpacingX,
                                     double gridSpacingY, double gridSpacingZ,
                                     double originX, double originY,
                                     double originZ, double initialValueU,
                                     double initialValueV, double initialValueW)
    : FaceCenteredGrid3(Size3(resolutionX, resolutionY, resolutionZ),
                        Vector3D(gridSpacingX, gridSpacingY, gridSpacingZ),
                        Vector3D(originX, originY, originZ),
                        Vector3D(initialValueU, initialValueV, initialValueW)) {
}

FaceCenteredGrid3::FaceCenteredGrid3(const Size3& resolution,
                                     const Vector3D& gridSpacing,
                                     const Vector3D& origin,
                                     const Vector3D& initialValue)
    : _uLinearSampler(LinearArraySampler3<double, double>(
          _dataU.constAccessor(), Vector3D(1, 1, 1), _dataOriginU)),
      _vLinearSampler(LinearArraySampler3<double, double>(
          _dataV.constAccessor(), Vector3D(1, 1, 1), _dataOriginV)),
      _wLinearSampler(LinearArraySampler3<double, double>(
          _dataW.constAccessor(), Vector3D(1, 1, 1), _dataOriginW)) {
    resize(resolution, gridSpacing, origin, initialValue);
}

FaceCenteredGrid3::FaceCenteredGrid3(const FaceCenteredGrid3& other)
    : _uLinearSampler(LinearArraySampler3<double, double>(
          _dataU.constAccessor(), Vector3D(1, 1, 1), _dataOriginU)),
      _vLinearSampler(LinearArraySampler3<double, double>(
          _dataV.constAccessor(), Vector3D(1, 1, 1), _dataOriginV)),
      _wLinearSampler(LinearArraySampler3<double, double>(
          _dataW.constAccessor(), Vector3D(1, 1, 1), _dataOriginW)) {
    set(other);
}

void FaceCenteredGrid3::swap(Grid3* other) {
    FaceCenteredGrid3* sameType = dynamic_cast<FaceCenteredGrid3*>(other);

    if (sameType != nullptr) {
        swapGrid(sameType);

        _dataU.swap(sameType->_dataU);
        _dataV.swap(sameType->_dataV);
        _dataW.swap(sameType->_dataW);
        std::swap(_dataOriginU, sameType->_dataOriginU);
        std::swap(_dataOriginV, sameType->_dataOriginV);
        std::swap(_dataOriginW, sameType->_dataOriginW);
        std::swap(_uLinearSampler, sameType->_uLinearSampler);
        std::swap(_vLinearSampler, sameType->_vLinearSampler);
        std::swap(_wLinearSampler, sameType->_wLinearSampler);
        std::swap(_sampler, sameType->_sampler);
    }
}

void FaceCenteredGrid3::set(const FaceCenteredGrid3& other) {
    setGrid(other);

    _dataU.set(other._dataU);
    _dataV.set(other._dataV);
    _dataW.set(other._dataW);
    _dataOriginU = other._dataOriginU;
    _dataOriginV = other._dataOriginV;
    _dataOriginW = other._dataOriginW;

    resetSampler();
}

FaceCenteredGrid3& FaceCenteredGrid3::operator=(
    const FaceCenteredGrid3& other) {
    set(other);
    return *this;
}

double& FaceCenteredGrid3::u(size_t i, size_t j, size_t k) {
    return _dataU(i, j, k);
}

const double& FaceCenteredGrid3::u(size_t i, size_t j, size_t k) const {
    return _dataU(i, j, k);
}

double& FaceCenteredGrid3::v(size_t i, size_t j, size_t k) {
    return _dataV(i, j, k);
}

const double& FaceCenteredGrid3::v(size_t i, size_t j, size_t k) const {
    return _dataV(i, j, k);
}

double& FaceCenteredGrid3::w(size_t i, size_t j, size_t k) {
    return _dataW(i, j, k);
}

const double& FaceCenteredGrid3::w(size_t i, size_t j, size_t k) const {
    return _dataW(i, j, k);
}

Vector3D FaceCenteredGrid3::valueAtCellCenter(size_t i, size_t j,
                                              size_t k) const {
    JET_ASSERT(i < resolution().x && j < resolution().y && k < resolution().z);

    return 0.5 * Vector3D(_dataU(i, j, k) + _dataU(i + 1, j, k),
                          _dataV(i, j, k) + _dataV(i, j + 1, k),
                          _dataW(i, j, k) + _dataW(i, j, k + 1));
}

double FaceCenteredGrid3::divergenceAtCellCenter(size_t i, size_t j,
                                                 size_t k) const {
    JET_ASSERT(i < resolution().x && j < resolution().y && k < resolution().z);

    const Vector3D& gs = gridSpacing();

    double leftU = _dataU(i, j, k);
    double rightU = _dataU(i + 1, j, k);
    double bottomV = _dataV(i, j, k);
    double topV = _dataV(i, j + 1, k);
    double backW = _dataW(i, j, k);
    double frontW = _dataW(i, j, k + 1);

    return (rightU - leftU) / gs.x + (topV - bottomV) / gs.y +
           (frontW - backW) / gs.z;
}

Vector3D FaceCenteredGrid3::curlAtCellCenter(size_t i, size_t j,
                                             size_t k) const {
    const Size3& res = resolution();
    const Vector3D& gs = gridSpacing();

    JET_ASSERT(i < res.x && j < res.y && k < res.z);

    Vector3D left = valueAtCellCenter((i > 0) ? i - 1 : i, j, k);
    Vector3D right = valueAtCellCenter((i + 1 < res.x) ? i + 1 : i, j, k);
    Vector3D down = valueAtCellCenter(i, (j > 0) ? j - 1 : j, k);
    Vector3D up = valueAtCellCenter(i, (j + 1 < res.y) ? j + 1 : j, k);
    Vector3D back = valueAtCellCenter(i, j, (k > 0) ? k - 1 : k);
    Vector3D front = valueAtCellCenter(i, j, (k + 1 < res.z) ? k + 1 : k);

    double Fx_ym = down.x;
    double Fx_yp = up.x;
    double Fx_zm = back.x;
    double Fx_zp = front.x;

    double Fy_xm = left.y;
    double Fy_xp = right.y;
    double Fy_zm = back.y;
    double Fy_zp = front.y;

    double Fz_xm = left.z;
    double Fz_xp = right.z;
    double Fz_ym = down.z;
    double Fz_yp = up.z;

    return Vector3D(
        0.5 * (Fz_yp - Fz_ym) / gs.y - 0.5 * (Fy_zp - Fy_zm) / gs.z,
        0.5 * (Fx_zp - Fx_zm) / gs.z - 0.5 * (Fz_xp - Fz_xm) / gs.x,
        0.5 * (Fy_xp - Fy_xm) / gs.x - 0.5 * (Fx_yp - Fx_ym) / gs.y);
}

FaceCenteredGrid3::ScalarDataAccessor FaceCenteredGrid3::uAccessor() {
    return _dataU.accessor();
}

FaceCenteredGrid3::ConstScalarDataAccessor FaceCenteredGrid3::uConstAccessor()
    const {
    return _dataU.constAccessor();
}

FaceCenteredGrid3::ScalarDataAccessor FaceCenteredGrid3::vAccessor() {
    return _dataV.accessor();
}

FaceCenteredGrid3::ConstScalarDataAccessor FaceCenteredGrid3::vConstAccessor()
    const {
    return _dataV.constAccessor();
}

FaceCenteredGrid3::ScalarDataAccessor FaceCenteredGrid3::wAccessor() {
    return _dataW.accessor();
}

FaceCenteredGrid3::ConstScalarDataAccessor FaceCenteredGrid3::wConstAccessor()
    const {
    return _dataW.constAccessor();
}

VectorGrid3::DataPositionFunc FaceCenteredGrid3::uPosition() const {
    Vector3D h = gridSpacing();

    return [this, h](size_t i, size_t j, size_t k) -> Vector3D {
        return _dataOriginU + h * Vector3D({i, j, k});
    };
}

VectorGrid3::DataPositionFunc FaceCenteredGrid3::vPosition() const {
    Vector3D h = gridSpacing();

    return [this, h](size_t i, size_t j, size_t k) -> Vector3D {
        return _dataOriginV + h * Vector3D({i, j, k});
    };
}

VectorGrid3::DataPositionFunc FaceCenteredGrid3::wPosition() const {
    Vector3D h = gridSpacing();

    return [this, h](size_t i, size_t j, size_t k) -> Vector3D {
        return _dataOriginW + h * Vector3D({i, j, k});
    };
}

Size3 FaceCenteredGrid3::uSize() const { return _dataU.size(); }

Size3 FaceCenteredGrid3::vSize() const { return _dataV.size(); }

Size3 FaceCenteredGrid3::wSize() const { return _dataW.size(); }

Vector3D FaceCenteredGrid3::uOrigin() const { return _dataOriginU; }

Vector3D FaceCenteredGrid3::vOrigin() const { return _dataOriginV; }

Vector3D FaceCenteredGrid3::wOrigin() const { return _dataOriginW; }

void FaceCenteredGrid3::fill(const Vector3D& value, ExecutionPolicy policy) {
    parallelFor(kZeroSize, _dataU.width(), kZeroSize, _dataU.height(),
                kZeroSize, _dataU.depth(),
                [this, value](size_t i, size_t j, size_t k) {
                    _dataU(i, j, k) = value.x;
                },
                policy);

    parallelFor(kZeroSize, _dataV.width(), kZeroSize, _dataV.height(),
                kZeroSize, _dataV.depth(),
                [this, value](size_t i, size_t j, size_t k) {
                    _dataV(i, j, k) = value.y;
                },
                policy);

    parallelFor(kZeroSize, _dataW.width(), kZeroSize, _dataW.height(),
                kZeroSize, _dataW.depth(),
                [this, value](size_t i, size_t j, size_t k) {
                    _dataW(i, j, k) = value.z;
                },
                policy);
}

void FaceCenteredGrid3::fill(
    const std::function<Vector3D(const Vector3D&)>& func,
    ExecutionPolicy policy) {
    DataPositionFunc uPos = uPosition();
    parallelFor(kZeroSize, _dataU.width(), kZeroSize, _dataU.height(),
                kZeroSize, _dataU.depth(),
                [this, &func, &uPos](size_t i, size_t j, size_t k) {
                    _dataU(i, j, k) = func(uPos(i, j, k)).x;
                },
                policy);
    DataPositionFunc vPos = vPosition();
    parallelFor(kZeroSize, _dataV.width(), kZeroSize, _dataV.height(),
                kZeroSize, _dataV.depth(),
                [this, &func, &vPos](size_t i, size_t j, size_t k) {
                    _dataV(i, j, k) = func(vPos(i, j, k)).y;
                },
                policy);
    DataPositionFunc wPos = wPosition();
    parallelFor(kZeroSize, _dataW.width(), kZeroSize, _dataW.height(),
                kZeroSize, _dataW.depth(),
                [this, &func, &wPos](size_t i, size_t j, size_t k) {
                    _dataW(i, j, k) = func(wPos(i, j, k)).z;
                },
                policy);
}

std::shared_ptr<VectorGrid3> FaceCenteredGrid3::clone() const {
    return CLONE_W_CUSTOM_DELETER(FaceCenteredGrid3);
}

void FaceCenteredGrid3::forEachUIndex(
    const std::function<void(size_t, size_t, size_t)>& func) const {
    _dataU.forEachIndex(func);
}

void FaceCenteredGrid3::parallelForEachUIndex(
    const std::function<void(size_t, size_t, size_t)>& func) const {
    _dataU.parallelForEachIndex(func);
}

void FaceCenteredGrid3::forEachVIndex(
    const std::function<void(size_t, size_t, size_t)>& func) const {
    _dataV.forEachIndex(func);
}

void FaceCenteredGrid3::parallelForEachVIndex(
    const std::function<void(size_t, size_t, size_t)>& func) const {
    _dataV.parallelForEachIndex(func);
}

void FaceCenteredGrid3::forEachWIndex(
    const std::function<void(size_t, size_t, size_t)>& func) const {
    _dataW.forEachIndex(func);
}

void FaceCenteredGrid3::parallelForEachWIndex(
    const std::function<void(size_t, size_t, size_t)>& func) const {
    _dataW.parallelForEachIndex(func);
}

Vector3D FaceCenteredGrid3::sample(const Vector3D& x) const {
    return _sampler(x);
}

std::function<Vector3D(const Vector3D&)> FaceCenteredGrid3::sampler() const {
    return _sampler;
}

double FaceCenteredGrid3::divergence(const Vector3D& x) const {
    Size3 res = resolution();
    ssize_t i, j, k;
    double fx, fy, fz;
    Vector3D cellCenterOrigin = origin() + 0.5 * gridSpacing();

    Vector3D normalizedX = (x - cellCenterOrigin) / gridSpacing();

    getBarycentric(normalizedX.x, 0, static_cast<ssize_t>(res.x) - 1, &i, &fx);
    getBarycentric(normalizedX.y, 0, static_cast<ssize_t>(res.y) - 1, &j, &fy);
    getBarycentric(normalizedX.z, 0, static_cast<ssize_t>(res.z) - 1, &k, &fz);

    std::array<Point3UI, 8> indices;
    std::array<double, 8> weights;

    indices[0] = Point3UI(i, j, k);
    indices[1] = Point3UI(i + 1, j, k);
    indices[2] = Point3UI(i, j + 1, k);
    indices[3] = Point3UI(i + 1, j + 1, k);
    indices[4] = Point3UI(i, j, k + 1);
    indices[5] = Point3UI(i + 1, j, k + 1);
    indices[6] = Point3UI(i, j + 1, k + 1);
    indices[7] = Point3UI(i + 1, j + 1, k + 1);

    weights[0] = (1.0 - fx) * (1.0 - fy) * (1.0 - fz);
    weights[1] = fx * (1.0 - fy) * (1.0 - fz);
    weights[2] = (1.0 - fx) * fy * (1.0 - fz);
    weights[3] = fx * fy * (1.0 - fz);
    weights[4] = (1.0 - fx) * (1.0 - fy) * fz;
    weights[5] = fx * (1.0 - fy) * fz;
    weights[6] = (1.0 - fx) * fy * fz;
    weights[7] = fx * fy * fz;

    double result = 0.0;

    for (int n = 0; n < 8; ++n) {
        result += weights[n] * divergenceAtCellCenter(
                                   indices[n].x, indices[n].y, indices[n].z);
    }

    return result;
}

Vector3D FaceCenteredGrid3::curl(const Vector3D& x) const {
    Size3 res = resolution();
    ssize_t i, j, k;
    double fx, fy, fz;
    Vector3D cellCenterOrigin = origin() + 0.5 * gridSpacing();

    Vector3D normalizedX = (x - cellCenterOrigin) / gridSpacing();

    getBarycentric(normalizedX.x, 0, static_cast<ssize_t>(res.x) - 1, &i, &fx);
    getBarycentric(normalizedX.y, 0, static_cast<ssize_t>(res.y) - 1, &j, &fy);
    getBarycentric(normalizedX.z, 0, static_cast<ssize_t>(res.z) - 1, &k, &fz);

    std::array<Point3UI, 8> indices;
    std::array<double, 8> weights;

    indices[0] = Point3UI(i, j, k);
    indices[1] = Point3UI(i + 1, j, k);
    indices[2] = Point3UI(i, j + 1, k);
    indices[3] = Point3UI(i + 1, j + 1, k);
    indices[4] = Point3UI(i, j, k + 1);
    indices[5] = Point3UI(i + 1, j, k + 1);
    indices[6] = Point3UI(i, j + 1, k + 1);
    indices[7] = Point3UI(i + 1, j + 1, k + 1);

    weights[0] = (1.0 - fx) * (1.0 - fy) * (1.0 - fz);
    weights[1] = fx * (1.0 - fy) * (1.0 - fz);
    weights[2] = (1.0 - fx) * fy * (1.0 - fz);
    weights[3] = fx * fy * (1.0 - fz);
    weights[4] = (1.0 - fx) * (1.0 - fy) * fz;
    weights[5] = fx * (1.0 - fy) * fz;
    weights[6] = (1.0 - fx) * fy * fz;
    weights[7] = fx * fy * fz;

    Vector3D result;

    for (int n = 0; n < 8; ++n) {
        result += weights[n] *
                  curlAtCellCenter(indices[n].x, indices[n].y, indices[n].z);
    }

    return result;
}

void FaceCenteredGrid3::onResize(const Size3& resolution,
                                 const Vector3D& gridSpacing,
                                 const Vector3D& origin,
                                 const Vector3D& initialValue) {
    if (resolution != Size3(0, 0, 0)) {
        _dataU.resize(resolution + Size3(1, 0, 0), initialValue.x);
        _dataV.resize(resolution + Size3(0, 1, 0), initialValue.y);
        _dataW.resize(resolution + Size3(0, 0, 1), initialValue.z);
    } else {
        _dataU.resize(Size3(0, 0, 0));
        _dataV.resize(Size3(0, 0, 0));
        _dataW.resize(Size3(0, 0, 0));
    }
    _dataOriginU = origin + 0.5 * Vector3D(0.0, gridSpacing.y, gridSpacing.z);
    _dataOriginV = origin + 0.5 * Vector3D(gridSpacing.x, 0.0, gridSpacing.z);
    _dataOriginW = origin + 0.5 * Vector3D(gridSpacing.x, gridSpacing.y, 0.0);

    resetSampler();
}

void FaceCenteredGrid3::resetSampler() {
    LinearArraySampler3<double, double> uSampler(_dataU.constAccessor(),
                                                 gridSpacing(), _dataOriginU);
    LinearArraySampler3<double, double> vSampler(_dataV.constAccessor(),
                                                 gridSpacing(), _dataOriginV);
    LinearArraySampler3<double, double> wSampler(_dataW.constAccessor(),
                                                 gridSpacing(), _dataOriginW);

    _uLinearSampler = uSampler;
    _vLinearSampler = vSampler;
    _wLinearSampler = wSampler;

    _sampler = [uSampler, vSampler, wSampler](const Vector3D& x) -> Vector3D {
        double u = uSampler(x);
        double v = vSampler(x);
        double w = wSampler(x);
        return Vector3D(u, v, w);
    };
}

FaceCenteredGrid3::Builder FaceCenteredGrid3::builder() { return Builder(); }

void FaceCenteredGrid3::getData(std::vector<double>* data) const {
    size_t size = uSize().x * uSize().y * uSize().z +
                  vSize().x * vSize().y * vSize().z +
                  wSize().x * wSize().y * wSize().z;
    data->resize(size);
    size_t cnt = 0;
    _dataU.forEach([&](double value) { (*data)[cnt++] = value; });
    _dataV.forEach([&](double value) { (*data)[cnt++] = value; });
    _dataW.forEach([&](double value) { (*data)[cnt++] = value; });
}

void FaceCenteredGrid3::setData(const std::vector<double>& data) {
    JET_ASSERT(uSize().x * uSize().y * uSize().z +
                   vSize().x * vSize().y * vSize().z +
                   wSize().x * wSize().y * wSize().z ==
               data.size());

    size_t cnt = 0;
    _dataU.forEachIndex(
        [&](size_t i, size_t j, size_t k) { _dataU(i, j, k) = data[cnt++]; });
    _dataV.forEachIndex(
        [&](size_t i, size_t j, size_t k) { _dataV(i, j, k) = data[cnt++]; });
    _dataW.forEachIndex(
        [&](size_t i, size_t j, size_t k) { _dataW(i, j, k) = data[cnt++]; });
}

FaceCenteredGrid3::Builder& FaceCenteredGrid3::Builder::withResolution(
    const Size3& resolution) {
    _resolution = resolution;
    return *this;
}

FaceCenteredGrid3::Builder& FaceCenteredGrid3::Builder::withResolution(
    size_t resolutionX, size_t resolutionY, size_t resolutionZ) {
    _resolution.x = resolutionX;
    _resolution.y = resolutionY;
    _resolution.z = resolutionZ;
    return *this;
}

FaceCenteredGrid3::Builder& FaceCenteredGrid3::Builder::withGridSpacing(
    const Vector3D& gridSpacing) {
    _gridSpacing = gridSpacing;
    return *this;
}

FaceCenteredGrid3::Builder& FaceCenteredGrid3::Builder::withGridSpacing(
    double gridSpacingX, double gridSpacingY, double gridSpacingZ) {
    _gridSpacing.x = gridSpacingX;
    _gridSpacing.y = gridSpacingY;
    _gridSpacing.z = gridSpacingZ;
    return *this;
}

FaceCenteredGrid3::Builder& FaceCenteredGrid3::Builder::withOrigin(
    const Vector3D& gridOrigin) {
    _gridOrigin = gridOrigin;
    return *this;
}

FaceCenteredGrid3::Builder& FaceCenteredGrid3::Builder::withOrigin(
    double gridOriginX, double gridOriginY, double gridOriginZ) {
    _gridOrigin.x = gridOriginX;
    _gridOrigin.y = gridOriginY;
    _gridOrigin.z = gridOriginZ;
    return *this;
}

FaceCenteredGrid3::Builder& FaceCenteredGrid3::Builder::withInitialValue(
    const Vector3D& initialVal) {
    _initialVal = initialVal;
    return *this;
}

FaceCenteredGrid3::Builder& FaceCenteredGrid3::Builder::withInitialValue(
    double initialValX, double initialValY, double initialValZ) {
    _initialVal.x = initialValX;
    _initialVal.y = initialValY;
    _initialVal.z = initialValZ;
    return *this;
}

FaceCenteredGrid3 FaceCenteredGrid3::Builder::build() const {
    return FaceCenteredGrid3(_resolution, _gridSpacing, _gridOrigin,
                             _initialVal);
}

FaceCenteredGrid3Ptr FaceCenteredGrid3::Builder::makeShared() const {
    return std::shared_ptr<FaceCenteredGrid3>(
        new FaceCenteredGrid3(_resolution, _gridSpacing, _gridOrigin,
                              _initialVal),
        [](FaceCenteredGrid3* obj) { delete obj; });
}

VectorGrid3Ptr FaceCenteredGrid3::Builder::build(
    const Size3& resolution, const Vector3D& gridSpacing,
    const Vector3D& gridOrigin, const Vector3D& initialVal) const {
    return std::shared_ptr<FaceCenteredGrid3>(
        new FaceCenteredGrid3(resolution, gridSpacing, gridOrigin, initialVal),
        [](FaceCenteredGrid3* obj) { delete obj; });
}
