// Copyright (c) 2018 Doyub Kim
//
// I am making my contributions/submissions to this project solely in my
// personal capacity and am not conveying any rights to any intellectual
// property of any third parties.

#include <pch.h>

#include <jet/array_samplers.h>
#include <jet/face_centered_grid.h>
#include <jet/parallel.h>

namespace jet {

namespace internal {

double divergence(const FaceCenteredGrid2 &grid, const Vector2D &x) {
    ssize_t i, j;
    double fx, fy;
    Vector2D cellCenterOrigin = grid.origin() + 0.5 * grid.gridSpacing();

    Vector2D normalizedX = elemDiv((x - cellCenterOrigin), grid.gridSpacing());

    getBarycentric(normalizedX.x, 0, static_cast<ssize_t>(grid.resolution().x),
                   i, fx);
    getBarycentric(normalizedX.y, 0, static_cast<ssize_t>(grid.resolution().y),
                   j, fy);

    std::array<Vector2UZ, 4> indices;
    std::array<double, 4> weights;

    indices[0] = Vector2UZ(i, j);
    indices[1] = Vector2UZ(i + 1, j);
    indices[2] = Vector2UZ(i, j + 1);
    indices[3] = Vector2UZ(i + 1, j + 1);

    weights[0] = (1.0 - fx) * (1.0 - fy);
    weights[1] = fx * (1.0 - fy);
    weights[2] = (1.0 - fx) * fy;
    weights[3] = fx * fy;

    double result = 0.0;

    for (int n = 0; n < 4; ++n) {
        result += weights[n] * grid.divergenceAtCellCenter(indices[n]);
    }

    return result;
}

double curl(const FaceCenteredGrid2 &grid, const Vector2UZ &idx) {
    size_t i = idx.x;
    size_t j = idx.y;
    const Vector2D &gs = grid.gridSpacing();
    const Vector2UZ &res = grid.resolution();

    JET_ASSERT(i < res.x && j < res.y);

    Vector2D left = grid.valueAtCellCenter((i > 0) ? i - 1 : i, j);
    Vector2D right = grid.valueAtCellCenter((i + 1 < res.x) ? i + 1 : i, j);
    Vector2D bottom = grid.valueAtCellCenter(i, (j > 0) ? j - 1 : j);
    Vector2D top = grid.valueAtCellCenter(i, (j + 1 < res.y) ? j + 1 : j);

    double Fx_ym = bottom.x;
    double Fx_yp = top.x;

    double Fy_xm = left.y;
    double Fy_xp = right.y;

    return 0.5 * (Fy_xp - Fy_xm) / gs.x - 0.5 * (Fx_yp - Fx_ym) / gs.y;
}

double curl(const FaceCenteredGrid2 &grid, const Vector2D &x) {
    ssize_t i, j;
    double fx, fy;
    Vector2D cellCenterOrigin = grid.origin() + 0.5 * grid.gridSpacing();

    Vector2D normalizedX = elemDiv((x - cellCenterOrigin), grid.gridSpacing());

    getBarycentric(normalizedX.x, static_cast<ssize_t>(grid.resolution().x), i,
                   fx);
    getBarycentric(normalizedX.y, static_cast<ssize_t>(grid.resolution().y), j,
                   fy);

    std::array<Vector2UZ, 4> indices;
    std::array<double, 4> weights;

    indices[0] = Vector2UZ(i, j);
    indices[1] = Vector2UZ(i + 1, j);
    indices[2] = Vector2UZ(i, j + 1);
    indices[3] = Vector2UZ(i + 1, j + 1);

    weights[0] = (1.0 - fx) * (1.0 - fy);
    weights[1] = fx * (1.0 - fy);
    weights[2] = (1.0 - fx) * fy;
    weights[3] = fx * fy;

    double result = 0.0;

    for (int n = 0; n < 4; ++n) {
        result += weights[n] * grid.curlAtCellCenter(indices[n]);
    }

    return result;
}

double divergence(const FaceCenteredGrid3 &grid, const Vector3D &x) {
    Vector3UZ res = grid.resolution();
    ssize_t i, j, k;
    double fx, fy, fz;
    Vector3D cellCenterOrigin = grid.origin() + 0.5 * grid.gridSpacing();

    Vector3D normalizedX = elemDiv((x - cellCenterOrigin), grid.gridSpacing());

    getBarycentric(normalizedX.x, static_cast<ssize_t>(res.x), i, fx);
    getBarycentric(normalizedX.y, static_cast<ssize_t>(res.y), j, fy);
    getBarycentric(normalizedX.z, static_cast<ssize_t>(res.z), k, fz);

    std::array<Vector3UZ, 8> indices;
    std::array<double, 8> weights;

    indices[0] = Vector3UZ(i, j, k);
    indices[1] = Vector3UZ(i + 1, j, k);
    indices[2] = Vector3UZ(i, j + 1, k);
    indices[3] = Vector3UZ(i + 1, j + 1, k);
    indices[4] = Vector3UZ(i, j, k + 1);
    indices[5] = Vector3UZ(i + 1, j, k + 1);
    indices[6] = Vector3UZ(i, j + 1, k + 1);
    indices[7] = Vector3UZ(i + 1, j + 1, k + 1);

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
        result += weights[n] * grid.divergenceAtCellCenter(
                                   indices[n].x, indices[n].y, indices[n].z);
    }

    return result;
}

Vector3D curl(const FaceCenteredGrid3 &grid, const Vector3UZ &idx) {
    size_t i = idx.x;
    size_t j = idx.y;
    size_t k = idx.z;
    const Vector3D &gs = grid.gridSpacing();
    const Vector3UZ &res = grid.resolution();

    JET_ASSERT(i < res.x && j < res.y && k < res.z);

    Vector3D left = grid.valueAtCellCenter((i > 0) ? i - 1 : i, j, k);
    Vector3D right = grid.valueAtCellCenter((i + 1 < res.x) ? i + 1 : i, j, k);
    Vector3D down = grid.valueAtCellCenter(i, (j > 0) ? j - 1 : j, k);
    Vector3D up = grid.valueAtCellCenter(i, (j + 1 < res.y) ? j + 1 : j, k);
    Vector3D back = grid.valueAtCellCenter(i, j, (k > 0) ? k - 1 : k);
    Vector3D front = grid.valueAtCellCenter(i, j, (k + 1 < res.z) ? k + 1 : k);

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

Vector3D curl(const FaceCenteredGrid3 &grid, const Vector3D &x) {
    Vector3UZ res = grid.resolution();
    ssize_t i, j, k;
    double fx, fy, fz;
    Vector3D cellCenterOrigin = grid.origin() + 0.5 * grid.gridSpacing();

    Vector3D normalizedX = elemDiv((x - cellCenterOrigin), grid.gridSpacing());

    getBarycentric(normalizedX.x, static_cast<ssize_t>(res.x), i, fx);
    getBarycentric(normalizedX.y, static_cast<ssize_t>(res.y), j, fy);
    getBarycentric(normalizedX.z, static_cast<ssize_t>(res.z), k, fz);

    std::array<Vector3UZ, 8> indices;
    std::array<double, 8> weights;

    indices[0] = Vector3UZ(i, j, k);
    indices[1] = Vector3UZ(i + 1, j, k);
    indices[2] = Vector3UZ(i, j + 1, k);
    indices[3] = Vector3UZ(i + 1, j + 1, k);
    indices[4] = Vector3UZ(i, j, k + 1);
    indices[5] = Vector3UZ(i + 1, j, k + 1);
    indices[6] = Vector3UZ(i, j + 1, k + 1);
    indices[7] = Vector3UZ(i + 1, j + 1, k + 1);

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
        result += weights[n] * grid.curlAtCellCenter(indices[n]);
    }

    return result;
}

}  // namespace internal

template <size_t N>
FaceCenteredGrid<N>::FaceCenteredGrid() {
    for (size_t i = 0; i < N; ++i) {
        Vector<double, N> dataOrigin = Vector<double, N>::makeConstant(0.5);
        dataOrigin[i] = 0.0;
        _dataOrigins[i] = dataOrigin;

        _linearSamplers[i] = LinearArraySampler<double, N>(
            _data[i], Vector<double, N>::makeConstant(1.0), dataOrigin);
    }
}

template <size_t N>
FaceCenteredGrid<N>::FaceCenteredGrid(const Vector<size_t, N> &resolution,
                                      const Vector<double, N> &gridSpacing,
                                      const Vector<double, N> &origin,
                                      const Vector<double, N> &initialValue)
    : FaceCenteredGrid() {
    resize(resolution, gridSpacing, origin, initialValue);
}

template <size_t N>
FaceCenteredGrid<N>::FaceCenteredGrid(const FaceCenteredGrid &other)
    : FaceCenteredGrid() {
    set(other);
}

template <size_t N>
void FaceCenteredGrid<N>::swap(Grid<N> *other) {
    FaceCenteredGrid *sameType = dynamic_cast<FaceCenteredGrid *>(other);

    if (sameType != nullptr) {
        swapGrid(sameType);

        for (size_t i = 0; i < N; ++i) {
            _data[i].swap(sameType->_data[i]);
            std::swap(_dataOrigins[i], sameType->_dataOrigins[i]);
            std::swap(_linearSamplers[i], sameType->_linearSamplers[i]);
        }

        std::swap(_sampler, sameType->_sampler);
    }
}

template <size_t N>
void FaceCenteredGrid<N>::set(const FaceCenteredGrid &other) {
    setGrid(other);

    for (size_t i = 0; i < N; ++i) {
        _data[i].copyFrom(other._data[i]);
    }

    _dataOrigins = other._dataOrigins;

    resetSampler();
}

template <size_t N>
FaceCenteredGrid<N> &FaceCenteredGrid<N>::operator=(
    const FaceCenteredGrid &other) {
    set(other);
    return *this;
}

template <size_t N>
double &FaceCenteredGrid<N>::u(const Vector<size_t, N> &idx) {
    return _data[0](idx);
}

template <size_t N>
const double &FaceCenteredGrid<N>::u(const Vector<size_t, N> &idx) const {
    return _data[0](idx);
}

template <size_t N>
double &FaceCenteredGrid<N>::v(const Vector<size_t, N> &idx) {
    return _data[1](idx);
}

template <size_t N>
const double &FaceCenteredGrid<N>::v(const Vector<size_t, N> &idx) const {
    return _data[1](idx);
}

template <size_t N>
Vector<double, N> FaceCenteredGrid<N>::valueAtCellCenter(
    const Vector<size_t, N> &idx) const {
    Vector<double, N> result;
    for (size_t i = 0; i < N; ++i) {
        JET_ASSERT(idx[i] < resolution()[i]);
        result[i] = 0.5 * (_data[i](idx) +
                           _data[i](idx + Vector<size_t, N>::makeUnit(i)));
    }
    return result;
}

template <size_t N>
double FaceCenteredGrid<N>::divergenceAtCellCenter(
    const Vector<size_t, N> &idx) const {
    const Vector<double, N> &gs = gridSpacing();

    double sum = 0.0;
    for (size_t i = 0; i < N; ++i) {
        JET_ASSERT(idx[i] < resolution()[i]);
        sum +=
            (_data[i](idx + Vector<size_t, N>::makeUnit(i)) - _data[i](idx)) /
            gs[i];
    }

    return sum;
}

template <size_t N>
typename GetCurl<N>::type FaceCenteredGrid<N>::curlAtCellCenter(
    const Vector<size_t, N> &idx) const {
    return internal::curl(*this, idx);
}

template <size_t N>
typename FaceCenteredGrid<N>::ScalarDataView FaceCenteredGrid<N>::uView() {
    return dataView(0);
}

template <size_t N>
typename FaceCenteredGrid<N>::ConstScalarDataView FaceCenteredGrid<N>::uView()
    const {
    return dataView(0);
}

template <size_t N>
typename FaceCenteredGrid<N>::ScalarDataView FaceCenteredGrid<N>::vView() {
    return dataView(1);
}

template <size_t N>
typename FaceCenteredGrid<N>::ConstScalarDataView FaceCenteredGrid<N>::vView()
    const {
    return dataView(1);
}

template <size_t N>
typename FaceCenteredGrid<N>::ScalarDataView FaceCenteredGrid<N>::dataView(
    size_t i) {
    return _data[i].view();
}

template <size_t N>
typename FaceCenteredGrid<N>::ConstScalarDataView FaceCenteredGrid<N>::dataView(
    size_t i) const {
    return _data[i].view();
}

template <size_t N>
GridDataPositionFunc<N> FaceCenteredGrid<N>::uPosition() const {
    return dataPosition(0);
}

template <size_t N>
GridDataPositionFunc<N> FaceCenteredGrid<N>::vPosition() const {
    return dataPosition(1);
}

template <size_t N>
GridDataPositionFunc<N> FaceCenteredGrid<N>::dataPosition(size_t i) const {
    Vector<double, N> h = gridSpacing();
    Vector<double, N> dataOrigin_ = _dataOrigins[i];

    return GridDataPositionFunc<N>(
        [h, dataOrigin_](const Vector<size_t, N> &idx) -> Vector<double, N> {
            return dataOrigin_ + elemMul(h, idx.template castTo<double>());
        });
}

template <size_t N>
Vector<size_t, N> FaceCenteredGrid<N>::uSize() const {
    return dataSize(0);
}

template <size_t N>
Vector<size_t, N> FaceCenteredGrid<N>::vSize() const {
    return dataSize(1);
}

template <size_t N>
Vector<size_t, N> FaceCenteredGrid<N>::dataSize(size_t i) const {
    return _data[i].size();
}

template <size_t N>
Vector<double, N> FaceCenteredGrid<N>::uOrigin() const {
    return dataOrigin(0);
}

template <size_t N>
Vector<double, N> FaceCenteredGrid<N>::vOrigin() const {
    return dataOrigin(1);
}

template <size_t N>
Vector<double, N> FaceCenteredGrid<N>::dataOrigin(size_t i) const {
    return _dataOrigins[i];
}

template <size_t N>
void FaceCenteredGrid<N>::fill(const Vector<double, N> &value,
                               ExecutionPolicy policy) {
    for (size_t i = 0; i < N; ++i) {
        parallelForEachIndex(dataSize(i),
                             [this, value, i](auto... indices) {
                                 _data[i](indices...) = value[i];
                             },
                             policy);
    }
}

template <size_t N>
void FaceCenteredGrid<N>::fill(
    const std::function<Vector<double, N>(const Vector<double, N> &)> &func,
    ExecutionPolicy policy) {
    for (size_t i = 0; i < N; ++i) {
        auto pos = dataPosition(i);
        parallelForEachIndex(dataSize(i),
                             [this, &func, &pos, i](auto... indices) {
                                 _data[i](indices...) =
                                     func(pos(indices...))[i];
                             },
                             policy);
    }
}

template <size_t N>
std::shared_ptr<VectorGrid<N>> FaceCenteredGrid<N>::clone() const {
    return CLONE_W_CUSTOM_DELETER(FaceCenteredGrid);
}

template <size_t N>
void FaceCenteredGrid<N>::forEachUIndex(
    const std::function<void(const Vector<size_t, N> &)> &func) const {
    forEachIndex(dataSize(0), GetUnroll<void, N>::unroll(func));
}

template <size_t N>
void FaceCenteredGrid<N>::parallelForEachUIndex(
    const std::function<void(const Vector<size_t, N> &)> &func) const {
    parallelForEachIndex(dataSize(0), GetUnroll<void, N>::unroll(func));
}

template <size_t N>
void FaceCenteredGrid<N>::forEachVIndex(
    const std::function<void(const Vector<size_t, N> &)> &func) const {
    forEachIndex(dataSize(1), GetUnroll<void, N>::unroll(func));
}

template <size_t N>
void FaceCenteredGrid<N>::parallelForEachVIndex(
    const std::function<void(const Vector<size_t, N> &)> &func) const {
    parallelForEachIndex(dataSize(1), GetUnroll<void, N>::unroll(func));
}

template <size_t N>
Vector<double, N> FaceCenteredGrid<N>::sample(
    const Vector<double, N> &x) const {
    return _sampler(x);
}

template <size_t N>
std::function<Vector<double, N>(const Vector<double, N> &)>
FaceCenteredGrid<N>::sampler() const {
    return _sampler;
}

template <size_t N>
double FaceCenteredGrid<N>::divergence(const Vector<double, N> &x) const {
    return internal::divergence(*this, x);
}

template <size_t N>
typename GetCurl<N>::type FaceCenteredGrid<N>::curl(
    const Vector<double, N> &x) const {
    return internal::curl(*this, x);
}

template <size_t N>
void FaceCenteredGrid<N>::onResize(const Vector<size_t, N> &resolution,
                                   const Vector<double, N> &gridSpacing,
                                   const Vector<double, N> &origin,
                                   const Vector<double, N> &initialValue) {
    for (size_t i = 0; i < N; ++i) {
        auto dataRes = (resolution != Vector<size_t, N>())
                           ? resolution + Vector<size_t, N>::makeUnit(i)
                           : resolution;
        _data[i].resize(dataRes, initialValue[i]);

        Vector<double, N> offset = 0.5 * gridSpacing;
        offset[i] = 0.0;
        _dataOrigins[i] = origin + offset;
    }

    resetSampler();
}

template <size_t N>
void FaceCenteredGrid<N>::resetSampler() {
    for (size_t i = 0; i < N; ++i) {
        _linearSamplers[i] = LinearArraySampler<double, N>(
            _data[i], gridSpacing(), _dataOrigins[i]);
    }

    auto linSamplers = _linearSamplers;

    _sampler = [linSamplers](const Vector<double, N> &x) -> Vector<double, N> {
        Vector<double, N> result;
        for (size_t i = 0; i < N; ++i) {
            result[i] = linSamplers[i](x);
        }
        return result;
    };
}

template <size_t N>
typename FaceCenteredGrid<N>::Builder FaceCenteredGrid<N>::builder() {
    return Builder();
}

template <size_t N>
void FaceCenteredGrid<N>::getData(Array1<double> &data) const {
    size_t size = 0;
    for (size_t i = 0; i < N; ++i) {
        size += product(dataSize(i), kOneSize);
    }

    data.resize(size);

    size_t cnt = 0;
    for (size_t i = 0; i < N; ++i) {
        std::for_each(_data[i].begin(), _data[i].end(),
                      [&](double value) { data[cnt++] = value; });
    }
}

template <size_t N>
void FaceCenteredGrid<N>::setData(const ConstArrayView1<double> &data) {
    size_t size = 0;
    for (size_t i = 0; i < N; ++i) {
        size += product(dataSize(i), kOneSize);
    }
    JET_ASSERT(size == data.length());

    size_t cnt = 0;
    for (size_t i = 0; i < N; ++i) {
        forEachIndex(_data[i].size(), [&](auto... indices) {
            _data[i](indices...) = data[cnt++];
        });
    }
}

template <size_t N>
typename FaceCenteredGrid<N>::Builder &
FaceCenteredGrid<N>::Builder::withResolution(
    const Vector<size_t, N> &resolution) {
    _resolution = resolution;
    return *this;
}

template <size_t N>
typename FaceCenteredGrid<N>::Builder &
FaceCenteredGrid<N>::Builder::withGridSpacing(
    const Vector<double, N> &gridSpacing) {
    _gridSpacing = gridSpacing;
    return *this;
}

template <size_t N>
typename FaceCenteredGrid<N>::Builder &FaceCenteredGrid<N>::Builder::withOrigin(
    const Vector<double, N> &gridOrigin) {
    _gridOrigin = gridOrigin;
    return *this;
}

template <size_t N>
typename FaceCenteredGrid<N>::Builder &
FaceCenteredGrid<N>::Builder::withInitialValue(
    const Vector<double, N> &initialVal) {
    _initialVal = initialVal;
    return *this;
}

template <size_t N>
FaceCenteredGrid<N> FaceCenteredGrid<N>::Builder::build() const {
    return FaceCenteredGrid(_resolution, _gridSpacing, _gridOrigin,
                            _initialVal);
}

template <size_t N>
std::shared_ptr<FaceCenteredGrid<N>> FaceCenteredGrid<N>::Builder::makeShared()
    const {
    return std::shared_ptr<FaceCenteredGrid>(
        new FaceCenteredGrid(_resolution, _gridSpacing, _gridOrigin,
                             _initialVal),
        [](FaceCenteredGrid *obj) { delete obj; });
}

template <size_t N>
std::shared_ptr<VectorGrid<N>> FaceCenteredGrid<N>::Builder::build(
    const Vector<size_t, N> &resolution, const Vector<double, N> &gridSpacing,
    const Vector<double, N> &gridOrigin,
    const Vector<double, N> &initialVal) const {
    return std::shared_ptr<FaceCenteredGrid>(
        new FaceCenteredGrid(resolution, gridSpacing, gridOrigin, initialVal),
        [](FaceCenteredGrid *obj) { delete obj; });
}

template class FaceCenteredGrid<2>;

template class FaceCenteredGrid<3>;

}  // namespace jet
