// Copyright (c) 2018 Doyub Kim
//
// I am making my contributions/submissions to this project solely in my
// personal capacity and am not conveying any rights to any intellectual
// property of any third parties.

#include <pch.h>
#include <jet/collocated_vector_grid3.h>
#include <jet/parallel.h>
#include <jet/serial.h>

#include <algorithm>
#include <utility>  // just make cpplint happy..
#include <vector>

using namespace jet;

CollocatedVectorGrid3::CollocatedVectorGrid3() :
    _linearSampler(_data.constAccessor(), Vector3D(1, 1, 1), Vector3D()) {
}

CollocatedVectorGrid3::~CollocatedVectorGrid3() {
}

const Vector3D& CollocatedVectorGrid3::operator()(
    size_t i, size_t j, size_t k) const {
    return _data(i, j, k);
}

Vector3D& CollocatedVectorGrid3::operator()(size_t i, size_t j, size_t k) {
    return _data(i, j, k);
}

double CollocatedVectorGrid3::divergenceAtDataPoint(
    size_t i, size_t j, size_t k) const {
    const Size3 ds = _data.size();
    const Vector3D& gs = gridSpacing();

    JET_ASSERT(i < ds.x && j < ds.y && k < ds.z);

    double left = _data((i > 0) ? i - 1 : i, j, k).x;
    double right = _data((i + 1 < ds.x) ? i + 1 : i, j, k).x;
    double down = _data(i, (j > 0) ? j - 1 : j, k).y;
    double up = _data(i, (j + 1 < ds.y) ? j + 1 : j, k).y;
    double back = _data(i, j, (k > 0) ? k - 1 : k).z;
    double front = _data(i, j, (k + 1 < ds.z) ? k + 1 : k).z;

    return 0.5 * (right - left) / gs.x
        + 0.5 * (up - down) / gs.y
        + 0.5 * (front - back) / gs.z;
}

Vector3D CollocatedVectorGrid3::curlAtDataPoint(
    size_t i, size_t j, size_t k) const {
    const Size3 ds = _data.size();
    const Vector3D& gs = gridSpacing();

    JET_ASSERT(i < ds.x && j < ds.y && k < ds.z);

    Vector3D left = _data((i > 0) ? i - 1 : i, j, k);
    Vector3D right = _data((i + 1 < ds.x) ? i + 1 : i, j, k);
    Vector3D down = _data(i, (j > 0) ? j - 1 : j, k);
    Vector3D up = _data(i, (j + 1 < ds.y) ? j + 1 : j, k);
    Vector3D back = _data(i, j, (k > 0) ? k - 1 : k);
    Vector3D front = _data(i, j, (k + 1 < ds.z) ? k + 1 : k);

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

Vector3D CollocatedVectorGrid3::sample(const Vector3D& x) const {
    return _sampler(x);
}

double CollocatedVectorGrid3::divergence(const Vector3D& x) const {
    std::array<Point3UI, 8> indices;
    std::array<double, 8> weights;
    _linearSampler.getCoordinatesAndWeights(x, &indices, &weights);

    double result = 0.0;

    for (int i = 0; i < 8; ++i) {
        result += weights[i] * divergenceAtDataPoint(
            indices[i].x, indices[i].y, indices[i].z);
    }

    return result;
}

Vector3D CollocatedVectorGrid3::curl(const Vector3D& x) const {
    std::array<Point3UI, 8> indices;
    std::array<double, 8> weights;
    _linearSampler.getCoordinatesAndWeights(x, &indices, &weights);

    Vector3D result;

    for (int i = 0; i < 8; ++i) {
        result += weights[i] * curlAtDataPoint(
            indices[i].x, indices[i].y, indices[i].z);
    }

    return result;
}

std::function<Vector3D(const Vector3D&)>
CollocatedVectorGrid3::sampler() const {
    return _sampler;
}

VectorGrid3::VectorDataAccessor CollocatedVectorGrid3::dataAccessor() {
    return _data.accessor();
}

VectorGrid3::ConstVectorDataAccessor
CollocatedVectorGrid3::constDataAccessor() const {
    return _data.constAccessor();
}

VectorGrid3::DataPositionFunc CollocatedVectorGrid3::dataPosition() const {
    Vector3D dataOrigin_ = dataOrigin();
    return [this, dataOrigin_](size_t i, size_t j, size_t k) -> Vector3D {
        return dataOrigin_ + gridSpacing() * Vector3D({i, j, k});
    };
}

void CollocatedVectorGrid3::forEachDataPointIndex(
    const std::function<void(size_t, size_t, size_t)>& func) const {
    _data.forEachIndex(func);
}

void CollocatedVectorGrid3::parallelForEachDataPointIndex(
    const std::function<void(size_t, size_t, size_t)>& func) const {
    _data.parallelForEachIndex(func);
}

void CollocatedVectorGrid3::swapCollocatedVectorGrid(
    CollocatedVectorGrid3* other) {
    swapGrid(other);

    _data.swap(other->_data);
    std::swap(_linearSampler, other->_linearSampler);
    std::swap(_sampler, other->_sampler);
}

void CollocatedVectorGrid3::setCollocatedVectorGrid(
    const CollocatedVectorGrid3& other) {
    setGrid(other);

    _data.set(other._data);
    resetSampler();
}

void CollocatedVectorGrid3::onResize(
    const Size3& resolution,
    const Vector3D& gridSpacing,
    const Vector3D& origin,
    const Vector3D& initialValue) {
    UNUSED_VARIABLE(resolution);
    UNUSED_VARIABLE(gridSpacing);
    UNUSED_VARIABLE(origin);

    _data.resize(dataSize(), initialValue);
    resetSampler();
}

void CollocatedVectorGrid3::resetSampler() {
    _linearSampler = LinearArraySampler3<Vector3D, double>(
        _data.constAccessor(), gridSpacing(), dataOrigin());
    _sampler = _linearSampler.functor();
}

void CollocatedVectorGrid3::getData(std::vector<double>* data) const {
    size_t size = 3 * dataSize().x * dataSize().y * dataSize().z;
    data->resize(size);
    size_t cnt = 0;
    _data.forEach([&] (const Vector3D& value) {
        (*data)[cnt++] = value.x;
        (*data)[cnt++] = value.y;
        (*data)[cnt++] = value.z;
    });
}

void CollocatedVectorGrid3::setData(const std::vector<double>& data) {
    JET_ASSERT(3 * dataSize().x * dataSize().y * dataSize().z == data.size());

    size_t cnt = 0;
    _data.forEachIndex([&] (size_t i, size_t j, size_t k) {
        _data(i, j, k).x = data[cnt++];
        _data(i, j, k).y = data[cnt++];
        _data(i, j, k).z = data[cnt++];
    });
}
