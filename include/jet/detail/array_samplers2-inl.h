// Copyright (c) 2018 Doyub Kim
//
// I am making my contributions/submissions to this project solely in my
// personal capacity and am not conveying any rights to any intellectual
// property of any third parties.

#ifndef INCLUDE_JET_DETAIL_ARRAY_SAMPLERS2_INL_H_
#define INCLUDE_JET_DETAIL_ARRAY_SAMPLERS2_INL_H_

#include <jet/macros.h>
#include <jet/math_utils.h>
#include <algorithm>
#include <limits>

namespace jet {

template <typename T, typename R>
NearestArraySampler2<T, R>::NearestArraySampler(
    const ConstArrayAccessor2<T>& accessor,
    const Vector2<R>& gridSpacing,
    const Vector2<R>& gridOrigin) {
    _gridSpacing = gridSpacing;
    _origin = gridOrigin;
    _accessor = accessor;
}

template <typename T, typename R>
NearestArraySampler2<T, R>::NearestArraySampler(
    const NearestArraySampler& other) {
    _gridSpacing = other._gridSpacing;
    _origin = other._origin;
    _accessor = other._accessor;
}

template <typename T, typename R>
T NearestArraySampler2<T, R>::operator()(const Vector2<R>& x) const {
    ssize_t i, j;
    R fx, fy;

    JET_ASSERT(_gridSpacing.x > std::numeric_limits<R>::epsilon() &&
               _gridSpacing.y > std::numeric_limits<R>::epsilon());
    Vector2<R> normalizedX = (x - _origin) / _gridSpacing;

    ssize_t iSize = static_cast<ssize_t>(_accessor.size().x);
    ssize_t jSize = static_cast<ssize_t>(_accessor.size().y);

    getBarycentric(normalizedX.x, 0, iSize - 1, &i, &fx);
    getBarycentric(normalizedX.y, 0, jSize - 1, &j, &fy);

    i = std::min(static_cast<ssize_t>(i + fx + 0.5), iSize - 1);
    j = std::min(static_cast<ssize_t>(j + fy + 0.5), jSize - 1);

    return _accessor(i, j);
}

template <typename T, typename R>
void NearestArraySampler2<T, R>::getCoordinate(
    const Vector2<R>& x, Point2UI* index) const {
    ssize_t i, j;
    R fx, fy;

    JET_ASSERT(_gridSpacing.x > std::numeric_limits<R>::epsilon() &&
               _gridSpacing.y > std::numeric_limits<R>::epsilon());
    Vector2<R> normalizedX = (x - _origin) / _gridSpacing;

    ssize_t iSize = static_cast<ssize_t>(_accessor.size().x);
    ssize_t jSize = static_cast<ssize_t>(_accessor.size().y);

    getBarycentric(normalizedX.x, 0, iSize - 1, &i, &fx);
    getBarycentric(normalizedX.y, 0, jSize - 1, &j, &fy);

    index->x = std::min(static_cast<ssize_t>(i + fx + 0.5), iSize - 1);
    index->y = std::min(static_cast<ssize_t>(j + fy + 0.5), jSize - 1);
}

template <typename T, typename R>
std::function<T(const Vector2<R>&)>
NearestArraySampler2<T, R>::functor() const {
    NearestArraySampler sampler(*this);
    return std::bind(
        &NearestArraySampler::operator(), sampler, std::placeholders::_1);
}

template <typename T, typename R>
LinearArraySampler2<T, R>::LinearArraySampler(
    const ConstArrayAccessor2<T>& accessor,
    const Vector2<R>& gridSpacing,
    const Vector2<R>& gridOrigin) {
    _gridSpacing = gridSpacing;
    _invGridSpacing = static_cast<R>(1) / _gridSpacing;
    _origin = gridOrigin;
    _accessor = accessor;
}

template <typename T, typename R>
LinearArraySampler2<T, R>::LinearArraySampler(
    const LinearArraySampler& other) {
    _gridSpacing = other._gridSpacing;
    _invGridSpacing = other._invGridSpacing;
    _origin = other._origin;
    _accessor = other._accessor;
}

template <typename T, typename R>
T LinearArraySampler2<T, R>::operator()(const Vector2<R>& x) const {
    ssize_t i, j;
    R fx, fy;

    JET_ASSERT(_gridSpacing.x > std::numeric_limits<R>::epsilon() &&
               _gridSpacing.y > std::numeric_limits<R>::epsilon());
    Vector2<R> normalizedX = (x - _origin) / _gridSpacing;

    ssize_t iSize = static_cast<ssize_t>(_accessor.size().x);
    ssize_t jSize = static_cast<ssize_t>(_accessor.size().y);

    getBarycentric(normalizedX.x, 0, iSize - 1, &i, &fx);
    getBarycentric(normalizedX.y, 0, jSize - 1, &j, &fy);

    ssize_t ip1 = std::min(i + 1, iSize - 1);
    ssize_t jp1 = std::min(j + 1, jSize - 1);

    return bilerp(
        _accessor(i, j),
        _accessor(ip1, j),
        _accessor(i, jp1),
        _accessor(ip1, jp1),
        fx,
        fy);
}

template <typename T, typename R>
void LinearArraySampler2<T, R>::getCoordinatesAndWeights(
    const Vector2<R>& x,
    std::array<Point2UI, 4>* indices,
    std::array<R, 4>* weights) const {
    ssize_t i, j;
    R fx, fy;

    JET_ASSERT(_gridSpacing.x > 0.0 && _gridSpacing.y > 0.0);

    Vector2<R> normalizedX = (x - _origin) / _gridSpacing;

    ssize_t iSize = static_cast<ssize_t>(_accessor.size().x);
    ssize_t jSize = static_cast<ssize_t>(_accessor.size().y);

    getBarycentric(normalizedX.x, 0, iSize - 1, &i, &fx);
    getBarycentric(normalizedX.y, 0, jSize - 1, &j, &fy);

    ssize_t ip1 = std::min(i + 1, iSize - 1);
    ssize_t jp1 = std::min(j + 1, jSize - 1);

    (*indices)[0] = Point2UI(i, j);
    (*indices)[1] = Point2UI(ip1, j);
    (*indices)[2] = Point2UI(i, jp1);
    (*indices)[3] = Point2UI(ip1, jp1);

    (*weights)[0] = (1 - fx) * (1 - fy);
    (*weights)[1] = fx * (1 - fy);
    (*weights)[2] = (1 - fx) * fy;
    (*weights)[3] = fx * fy;
}

template <typename T, typename R>
void LinearArraySampler2<T, R>::getCoordinatesAndGradientWeights(
    const Vector2<R>& x,
    std::array<Point2UI, 4>* indices,
    std::array<Vector2<R>, 4>* weights) const {
    ssize_t i, j;
    R fx, fy;

    JET_ASSERT(_gridSpacing.x > 0.0 && _gridSpacing.y > 0.0);

    const Vector2<R> normalizedX = (x - _origin) * _invGridSpacing;

    const ssize_t iSize = static_cast<ssize_t>(_accessor.size().x);
    const ssize_t jSize = static_cast<ssize_t>(_accessor.size().y);

    getBarycentric(normalizedX.x, 0, iSize - 1, &i, &fx);
    getBarycentric(normalizedX.y, 0, jSize - 1, &j, &fy);

    const ssize_t ip1 = std::min(i + 1, iSize - 1);
    const ssize_t jp1 = std::min(j + 1, jSize - 1);

    (*indices)[0] = Point2UI(i, j);
    (*indices)[1] = Point2UI(ip1, j);
    (*indices)[2] = Point2UI(i, jp1);
    (*indices)[3] = Point2UI(ip1, jp1);

    (*weights)[0] = Vector2<R>(
        fy * _invGridSpacing.x - _invGridSpacing.x,
        fx * _invGridSpacing.y - _invGridSpacing.y);
    (*weights)[1] = Vector2<R>(
        -fy * _invGridSpacing.x + _invGridSpacing.x,
        -fx * _invGridSpacing.y);
    (*weights)[2] = Vector2<R>(
        -fy * _invGridSpacing.x,
        -fx * _invGridSpacing.y + _invGridSpacing.y);
    (*weights)[3] = Vector2<R>(
        fy * _invGridSpacing.x,
        fx * _invGridSpacing.y);
}

template <typename T, typename R>
std::function<T(const Vector2<R>&)> LinearArraySampler2<T, R>::functor() const {
    LinearArraySampler sampler(*this);
    return std::bind(
        &LinearArraySampler::operator(), sampler, std::placeholders::_1);
}


template <typename T, typename R>
CubicArraySampler2<T, R>::CubicArraySampler(
    const ConstArrayAccessor2<T>& accessor,
    const Vector2<R>& gridSpacing,
    const Vector2<R>& gridOrigin) {
    _gridSpacing = gridSpacing;
    _origin = gridOrigin;
    _accessor = accessor;
}

template <typename T, typename R>
CubicArraySampler2<T, R>::CubicArraySampler(
    const CubicArraySampler& other) {
    _gridSpacing = other._gridSpacing;
    _origin = other._origin;
    _accessor = other._accessor;
}

template <typename T, typename R>
T CubicArraySampler2<T, R>::operator()(const Vector2<R>& x) const {
    ssize_t i, j;
    const ssize_t iSize = static_cast<ssize_t>(_accessor.size().x);
    const ssize_t jSize = static_cast<ssize_t>(_accessor.size().y);
    R fx, fy;

    JET_ASSERT(_gridSpacing.x > std::numeric_limits<R>::epsilon() &&
               _gridSpacing.y > std::numeric_limits<R>::epsilon());
    const Vector2<R> normalizedX = (x - _origin) / _gridSpacing;

    getBarycentric(normalizedX.x, 0, iSize - 1, &i, &fx);
    getBarycentric(normalizedX.y, 0, jSize - 1, &j, &fy);

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

    // Calculate in i direction first
    T values[4];
    for (int n = 0; n < 4; ++n) {
        values[n] = monotonicCatmullRom(
            _accessor(is[0], js[n]),
            _accessor(is[1], js[n]),
            _accessor(is[2], js[n]),
            _accessor(is[3], js[n]),
            fx);
    }

    return monotonicCatmullRom(values[0], values[1], values[2], values[3], fy);
}

template <typename T, typename R>
std::function<T(const Vector2<R>&)> CubicArraySampler2<T, R>::functor() const {
    CubicArraySampler sampler(*this);
    return std::bind(
        &CubicArraySampler::operator(), sampler, std::placeholders::_1);
}

}  // namespace jet

#endif  // INCLUDE_JET_DETAIL_ARRAY_SAMPLERS2_INL_H_
