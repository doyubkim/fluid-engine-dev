// Copyright (c) 2018 Doyub Kim
//
// I am making my contributions/submissions to this project solely in my
// personal capacity and am not conveying any rights to any intellectual
// property of any third parties.

#ifndef INCLUDE_JET_DETAIL_ARRAY_SAMPLERS1_INL_H_
#define INCLUDE_JET_DETAIL_ARRAY_SAMPLERS1_INL_H_

#include <jet/macros.h>
#include <jet/math_utils.h>
#include <algorithm>
#include <limits>

namespace jet {

template <typename T, typename R>
NearestArraySampler1<T, R>::NearestArraySampler(
    const ConstArrayAccessor1<T>& accessor,
    R gridSpacing,
    R gridOrigin) {
    _gridSpacing = gridSpacing;
    _origin = gridOrigin;
    _accessor = accessor;
}

template <typename T, typename R>
NearestArraySampler1<T, R>::NearestArraySampler(
    const NearestArraySampler& other) {
    _gridSpacing = other._gridSpacing;
    _origin = other._origin;
    _accessor = other._accessor;
}

template <typename T, typename R>
T NearestArraySampler1<T, R>::operator()(R x) const {
    ssize_t i;
    R fx;

    assert(_gridSpacing > std::numeric_limits<R>::epsilon());
    R normalizedX = (x - _origin) / _gridSpacing;

    ssize_t iSize = static_cast<ssize_t>(_accessor.size());

    getBarycentric(normalizedX, 0, iSize - 1, &i, &fx);

    i = std::min(static_cast<ssize_t>(i + fx + 0.5), iSize - 1);

    return _accessor[i];
}

template <typename T, typename R>
void NearestArraySampler1<T, R>::getCoordinate(R x, size_t* i) const {
    R fx;

    JET_ASSERT(_gridSpacing > std::numeric_limits<R>::epsilon());
    R normalizedX = (x - _origin) / _gridSpacing;

    ssize_t iSize = static_cast<ssize_t>(_accessor.size());

    ssize_t _i;
    getBarycentric(normalizedX, 0, iSize - 1, &_i, &fx);

    *i = std::min(static_cast<ssize_t>(_i + fx + 0.5), iSize - 1);
}

template <typename T, typename R>
std::function<T(R)> NearestArraySampler1<T, R>::functor() const {
    NearestArraySampler sampler(*this);
    return std::bind(
        &NearestArraySampler::operator(), sampler, std::placeholders::_1);
}

template <typename T, typename R>
LinearArraySampler1<T, R>::LinearArraySampler(
    const ConstArrayAccessor1<T>& accessor,
    R gridSpacing,
    R gridOrigin) {
    _gridSpacing = gridSpacing;
    _origin = gridOrigin;
    _accessor = accessor;
}

template <typename T, typename R>
LinearArraySampler1<T, R>::LinearArraySampler(
    const LinearArraySampler& other) {
    _gridSpacing = other._gridSpacing;
    _origin = other._origin;
    _accessor = other._accessor;
}

template <typename T, typename R>
T LinearArraySampler1<T, R>::operator()(R x) const {
    ssize_t i;
    R fx;

    assert(_gridSpacing > std::numeric_limits<R>::epsilon());
    R normalizedX = (x - _origin) / _gridSpacing;

    ssize_t iSize = static_cast<ssize_t>(_accessor.size());

    getBarycentric(normalizedX, 0, iSize - 1, &i, &fx);

    ssize_t ip1 = std::min(i + 1, iSize - 1);

    return lerp(
        _accessor[i],
        _accessor[ip1],
        fx);
}

template <typename T, typename R>
void LinearArraySampler1<T, R>::getCoordinatesAndWeights(
    R x, size_t* i0, size_t* i1, T* weight0, T* weight1) const {
    ssize_t i;
    R fx;

    JET_ASSERT(_gridSpacing > std::numeric_limits<R>::epsilon());
    R normalizedX = (x - _origin) / _gridSpacing;

    ssize_t iSize = static_cast<ssize_t>(_accessor.size().x);

    getBarycentric(normalizedX, 0, iSize - 1, &i, &fx);

    ssize_t ip1 = std::min(i + 1, iSize - 1);

    *i0 = i;
    *i1 = ip1;
    *weight0 = 1 - fx;
    *weight1 = fx;
}

template <typename T, typename R>
std::function<T(R)> LinearArraySampler1<T, R>::functor() const {
    LinearArraySampler sampler(*this);
    return std::bind(
        &LinearArraySampler::operator(), sampler, std::placeholders::_1);
}


template <typename T, typename R>
CubicArraySampler1<T, R>::CubicArraySampler(
    const ConstArrayAccessor1<T>& accessor,
    R gridSpacing,
    R gridOrigin) {
    _gridSpacing = gridSpacing;
    _origin = gridOrigin;
    _accessor = accessor;
}

template <typename T, typename R>
CubicArraySampler1<T, R>::CubicArraySampler(
    const CubicArraySampler& other) {
    _gridSpacing = other._gridSpacing;
    _origin = other._origin;
    _accessor = other._accessor;
}

template <typename T, typename R>
T CubicArraySampler1<T, R>::operator()(R x) const {
    ssize_t i;
    ssize_t iSize = static_cast<ssize_t>(_accessor.size());
    R fx;

    JET_ASSERT(_gridSpacing > std::numeric_limits<R>::epsilon());
    R normalizedX = (x - _origin) / _gridSpacing;

    getBarycentric(normalizedX, 0, iSize - 1, &i, &fx);

    ssize_t im1 = std::max(i - 1, kZeroSSize);
    ssize_t ip1 = std::min(i + 1, iSize - 1);
    ssize_t ip2 = std::min(i + 2, iSize - 1);

    return monotonicCatmullRom(
        _accessor[im1],
        _accessor[i],
        _accessor[ip1],
        _accessor[ip2],
        fx);
}

template <typename T, typename R>
std::function<T(R)> CubicArraySampler1<T, R>::functor() const {
    CubicArraySampler sampler(*this);
    return std::bind(
        &CubicArraySampler::operator(), sampler, std::placeholders::_1);
}

}  // namespace jet

#endif  // INCLUDE_JET_DETAIL_ARRAY_SAMPLERS1_INL_H_
