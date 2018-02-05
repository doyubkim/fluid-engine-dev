// Copyright (c) 2018 Doyub Kim
//
// I am making my contributions/submissions to this project solely in my
// personal capacity and am not conveying any rights to any intellectual
// property of any third parties.

#ifndef INCLUDE_JET_DETAIL_ARRAY_SAMPLERS3_INL_H_
#define INCLUDE_JET_DETAIL_ARRAY_SAMPLERS3_INL_H_

#include <jet/macros.h>
#include <jet/math_utils.h>
#include <algorithm>
#include <functional>
#include <limits>

namespace jet {

template <typename T, typename R>
NearestArraySampler3<T, R>::NearestArraySampler(
    const ConstArrayAccessor3<T>& accessor,
    const Vector3<R>& gridSpacing,
    const Vector3<R>& gridOrigin) {
    _gridSpacing = gridSpacing;
    _origin = gridOrigin;
    _accessor = accessor;
}

template <typename T, typename R>
NearestArraySampler3<T, R>::NearestArraySampler(
    const NearestArraySampler& other) {
    _gridSpacing = other._gridSpacing;
    _origin = other._origin;
    _accessor = other._accessor;
}

template <typename T, typename R>
T NearestArraySampler3<T, R>::operator()(const Vector3<R>& x) const {
    ssize_t i, j, k;
    R fx, fy, fz;

    JET_ASSERT(_gridSpacing.x > std::numeric_limits<R>::epsilon() &&
               _gridSpacing.y > std::numeric_limits<R>::epsilon() &&
               _gridSpacing.z > std::numeric_limits<R>::epsilon());
    Vector3<R> normalizedX = (x - _origin) / _gridSpacing;

    ssize_t iSize = static_cast<ssize_t>(_accessor.size().x);
    ssize_t jSize = static_cast<ssize_t>(_accessor.size().y);
    ssize_t kSize = static_cast<ssize_t>(_accessor.size().z);

    getBarycentric(normalizedX.x, 0, iSize - 1, &i, &fx);
    getBarycentric(normalizedX.y, 0, jSize - 1, &j, &fy);
    getBarycentric(normalizedX.z, 0, kSize - 1, &k, &fz);

    i = std::min(static_cast<ssize_t>(i + fx + 0.5), iSize - 1);
    j = std::min(static_cast<ssize_t>(j + fy + 0.5), jSize - 1);
    k = std::min(static_cast<ssize_t>(k + fz + 0.5), kSize - 1);

    return _accessor(i, j, k);
}

template <typename T, typename R>
void NearestArraySampler3<T, R>::getCoordinate(
    const Vector3<R>& x, Point3UI* index) const {
    ssize_t i, j, k;
    R fx, fy, fz;

    JET_ASSERT(_gridSpacing.x > std::numeric_limits<R>::epsilon() &&
               _gridSpacing.y > std::numeric_limits<R>::epsilon() &&
               _gridSpacing.z > std::numeric_limits<R>::epsilon());
    Vector3<R> normalizedX = (x - _origin) / _gridSpacing;

    ssize_t iSize = static_cast<ssize_t>(_accessor.size().x);
    ssize_t jSize = static_cast<ssize_t>(_accessor.size().y);
    ssize_t kSize = static_cast<ssize_t>(_accessor.size().z);

    getBarycentric(normalizedX.x, 0, iSize - 1, &i, &fx);
    getBarycentric(normalizedX.y, 0, jSize - 1, &j, &fy);
    getBarycentric(normalizedX.z, 0, kSize - 1, &k, &fz);

    index->x = std::min(static_cast<ssize_t>(i + fx + 0.5), iSize - 1);
    index->y = std::min(static_cast<ssize_t>(j + fy + 0.5), jSize - 1);
    index->z = std::min(static_cast<ssize_t>(k + fz + 0.5), kSize - 1);
}

template <typename T, typename R>
std::function<T(const Vector3<R>&)>

NearestArraySampler3<T, R>::functor() const {
    NearestArraySampler sampler(*this);
    return std::bind(
        &NearestArraySampler::operator(), sampler, std::placeholders::_1);
}

template <typename T, typename R>
LinearArraySampler3<T, R>::LinearArraySampler(
    const ConstArrayAccessor3<T>& accessor,
    const Vector3<R>& gridSpacing,
    const Vector3<R>& gridOrigin) {
    _gridSpacing = gridSpacing;
    _invGridSpacing = static_cast<R>(1) / _gridSpacing;
    _origin = gridOrigin;
    _accessor = accessor;
}


template <typename T, typename R>
LinearArraySampler3<T, R>::LinearArraySampler(
    const LinearArraySampler& other) {
    _gridSpacing = other._gridSpacing;
    _invGridSpacing = other._invGridSpacing;
    _origin = other._origin;
    _accessor = other._accessor;
}

template <typename T, typename R>
T LinearArraySampler3<T, R>::operator()(const Vector3<R>& x) const {
    ssize_t i, j, k;
    R fx, fy, fz;

    JET_ASSERT(_gridSpacing.x > std::numeric_limits<R>::epsilon() &&
               _gridSpacing.y > std::numeric_limits<R>::epsilon() &&
               _gridSpacing.z > std::numeric_limits<R>::epsilon());
    Vector3<R> normalizedX = (x - _origin) / _gridSpacing;

    ssize_t iSize = static_cast<ssize_t>(_accessor.size().x);
    ssize_t jSize = static_cast<ssize_t>(_accessor.size().y);
    ssize_t kSize = static_cast<ssize_t>(_accessor.size().z);

    getBarycentric(normalizedX.x, 0, iSize - 1, &i, &fx);
    getBarycentric(normalizedX.y, 0, jSize - 1, &j, &fy);
    getBarycentric(normalizedX.z, 0, kSize - 1, &k, &fz);

    ssize_t ip1 = std::min(i + 1, iSize - 1);
    ssize_t jp1 = std::min(j + 1, jSize - 1);
    ssize_t kp1 = std::min(k + 1, kSize - 1);

    return trilerp(
        _accessor(i, j, k),
        _accessor(ip1, j, k),
        _accessor(i, jp1, k),
        _accessor(ip1, jp1, k),
        _accessor(i, j, kp1),
        _accessor(ip1, j, kp1),
        _accessor(i, jp1, kp1),
        _accessor(ip1, jp1, kp1),
        fx,
        fy,
        fz);
}

template <typename T, typename R>
void LinearArraySampler3<T, R>::getCoordinatesAndWeights(
    const Vector3<R>& x,
    std::array<Point3UI, 8>* indices,
    std::array<R, 8>* weights) const {
    ssize_t i, j, k;
    R fx, fy, fz;

    JET_ASSERT(
        _gridSpacing.x > 0.0 && _gridSpacing.y > 0.0 && _gridSpacing.z > 0.0);

    const Vector3<R> normalizedX = (x - _origin) * _invGridSpacing;

    const ssize_t iSize = static_cast<ssize_t>(_accessor.size().x);
    const ssize_t jSize = static_cast<ssize_t>(_accessor.size().y);
    const ssize_t kSize = static_cast<ssize_t>(_accessor.size().z);

    getBarycentric(normalizedX.x, 0, iSize - 1, &i, &fx);
    getBarycentric(normalizedX.y, 0, jSize - 1, &j, &fy);
    getBarycentric(normalizedX.z, 0, kSize - 1, &k, &fz);

    const ssize_t ip1 = std::min(i + 1, iSize - 1);
    const ssize_t jp1 = std::min(j + 1, jSize - 1);
    const ssize_t kp1 = std::min(k + 1, kSize - 1);

    (*indices)[0] = Point3UI(i, j, k);
    (*indices)[1] = Point3UI(ip1, j, k);
    (*indices)[2] = Point3UI(i, jp1, k);
    (*indices)[3] = Point3UI(ip1, jp1, k);
    (*indices)[4] = Point3UI(i, j, kp1);
    (*indices)[5] = Point3UI(ip1, j, kp1);
    (*indices)[6] = Point3UI(i, jp1, kp1);
    (*indices)[7] = Point3UI(ip1, jp1, kp1);

    (*weights)[0] = (1 - fx) * (1 - fy) * (1 - fz);
    (*weights)[1] = fx * (1 - fy) * (1 - fz);
    (*weights)[2] = (1 - fx) * fy * (1 - fz);
    (*weights)[3] = fx * fy * (1 - fz);
    (*weights)[4] = (1 - fx) * (1 - fy) * fz;
    (*weights)[5] = fx * (1 - fy) * fz;
    (*weights)[6] = (1 - fx) * fy * fz;
    (*weights)[7] = fx * fy * fz;
}

template <typename T, typename R>
void LinearArraySampler3<T, R>::getCoordinatesAndGradientWeights(
    const Vector3<R>& x,
    std::array<Point3UI, 8>* indices,
    std::array<Vector3<R>, 8>* weights) const {
    ssize_t i, j, k;
    R fx, fy, fz;

    JET_ASSERT(
        _gridSpacing.x > 0.0 && _gridSpacing.y > 0.0 && _gridSpacing.z > 0.0);

    Vector3<R> normalizedX = (x - _origin) / _gridSpacing;

    ssize_t iSize = static_cast<ssize_t>(_accessor.size().x);
    ssize_t jSize = static_cast<ssize_t>(_accessor.size().y);
    ssize_t kSize = static_cast<ssize_t>(_accessor.size().z);

    getBarycentric(normalizedX.x, 0, iSize - 1, &i, &fx);
    getBarycentric(normalizedX.y, 0, jSize - 1, &j, &fy);
    getBarycentric(normalizedX.z, 0, kSize - 1, &k, &fz);

    ssize_t ip1 = std::min(i + 1, iSize - 1);
    ssize_t jp1 = std::min(j + 1, jSize - 1);
    ssize_t kp1 = std::min(k + 1, kSize - 1);

    (*indices)[0] = Point3UI(i, j, k);
    (*indices)[1] = Point3UI(ip1, j, k);
    (*indices)[2] = Point3UI(i, jp1, k);
    (*indices)[3] = Point3UI(ip1, jp1, k);
    (*indices)[4] = Point3UI(i, j, kp1);
    (*indices)[5] = Point3UI(ip1, j, kp1);
    (*indices)[6] = Point3UI(i, jp1, kp1);
    (*indices)[7] = Point3UI(ip1, jp1, kp1);

    (*weights)[0] = Vector3<R>(
        -_invGridSpacing.x * (1 - fy) * (1 - fz),
        -_invGridSpacing.y * (1 - fx) * (1 - fz),
        -_invGridSpacing.z * (1 - fx) * (1 - fy));
    (*weights)[1] = Vector3<R>(
        _invGridSpacing.x * (1 - fy) * (1 - fz),
        fx * (-_invGridSpacing.y) * (1 - fz),
        fx * (1 - fy) * (-_invGridSpacing.z));
    (*weights)[2] = Vector3<R>(
        (-_invGridSpacing.x) * fy * (1 - fz),
        (1 - fx) * _invGridSpacing.y * (1 - fz),
        (1 - fx) * fy * (-_invGridSpacing.z));
    (*weights)[3] = Vector3<R>(
        _invGridSpacing.x * fy * (1 - fz),
        fx * _invGridSpacing.y * (1 - fz),
        fx * fy * (-_invGridSpacing.z));
    (*weights)[4] = Vector3<R>(
        (-_invGridSpacing.x) * (1 - fy) * fz,
        (1 - fx) * (-_invGridSpacing.y) * fz,
        (1 - fx) * (1 - fy) * _invGridSpacing.z);
    (*weights)[5] = Vector3<R>(
        _invGridSpacing.x * (1 - fy) * fz,
        fx * (-_invGridSpacing.y) * fz,
        fx * (1 - fy) * _invGridSpacing.z);
    (*weights)[6] = Vector3<R>(
        (-_invGridSpacing.x) * fy * fz,
        (1 - fx) * _invGridSpacing.y * fz,
        (1 - fx) * fy * _invGridSpacing.z);
    (*weights)[7] = Vector3<R>(
        _invGridSpacing.x * fy * fz,
        fx * _invGridSpacing.y * fz,
        fx * fy * _invGridSpacing.z);
}

template <typename T, typename R>
std::function<T(const Vector3<R>&)> LinearArraySampler3<T, R>::functor() const {
    LinearArraySampler sampler(*this);
    return std::bind(
        &LinearArraySampler::operator(), sampler, std::placeholders::_1);
}


template <typename T, typename R>
CubicArraySampler3<T, R>::CubicArraySampler(
    const ConstArrayAccessor3<T>& accessor,
    const Vector3<R>& gridSpacing,
    const Vector3<R>& gridOrigin) {
    _gridSpacing = gridSpacing;
    _origin = gridOrigin;
    _accessor = accessor;
}


template <typename T, typename R>
CubicArraySampler3<T, R>::CubicArraySampler(
    const CubicArraySampler& other) {
    _gridSpacing = other._gridSpacing;
    _origin = other._origin;
    _accessor = other._accessor;
}

template <typename T, typename R>
T CubicArraySampler3<T, R>::operator()(const Vector3<R>& x) const {
    ssize_t i, j, k;
    ssize_t iSize = static_cast<ssize_t>(_accessor.size().x);
    ssize_t jSize = static_cast<ssize_t>(_accessor.size().y);
    ssize_t kSize = static_cast<ssize_t>(_accessor.size().z);
    R fx, fy, fz;

    JET_ASSERT(_gridSpacing.x > std::numeric_limits<R>::epsilon() &&
               _gridSpacing.y > std::numeric_limits<R>::epsilon() &&
               _gridSpacing.z > std::numeric_limits<R>::epsilon());
    Vector3<R> normalizedX = (x - _origin) / _gridSpacing;

    getBarycentric(normalizedX.x, 0, iSize - 1, &i, &fx);
    getBarycentric(normalizedX.y, 0, jSize - 1, &j, &fy);
    getBarycentric(normalizedX.z, 0, kSize - 1, &k, &fz);

    ssize_t is[4] = {
        std::max(i - 1, kZeroSSize),
        i,
        std::min(i + 1, iSize - 1),
        std::min(i + 2, iSize - 1)
    };
    ssize_t js[4] = {
        std::max(j - 1, kZeroSSize),
        j,
        std::min(j + 1, jSize - 1),
        std::min(j + 2, jSize - 1)
    };
    ssize_t ks[4] = {
        std::max(k - 1, kZeroSSize),
        k,
        std::min(k + 1, kSize - 1),
        std::min(k + 2, kSize - 1)
    };

    T kValues[4];

    for (int kk = 0; kk < 4; ++kk) {
        T jValues[4];

        for (int jj = 0; jj < 4; ++jj) {
            jValues[jj] = monotonicCatmullRom(
                _accessor(is[0], js[jj], ks[kk]),
                _accessor(is[1], js[jj], ks[kk]),
                _accessor(is[2], js[jj], ks[kk]),
                _accessor(is[3], js[jj], ks[kk]),
                fx);
        }

        kValues[kk] = monotonicCatmullRom(
            jValues[0], jValues[1], jValues[2], jValues[3], fy);
    }

    return monotonicCatmullRom(
        kValues[0], kValues[1], kValues[2], kValues[3], fz);
}

template <typename T, typename R>
std::function<T(const Vector3<R>&)> CubicArraySampler3<T, R>::functor() const {
    CubicArraySampler sampler(*this);
    return std::bind(
        &CubicArraySampler::operator(), sampler, std::placeholders::_1);
}

}  // namespace jet

#endif  // INCLUDE_JET_DETAIL_ARRAY_SAMPLERS3_INL_H_
