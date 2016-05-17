// Copyright (c) 2016 Doyub Kim

#ifndef INCLUDE_JET_MATH_UTILS_H_
#define INCLUDE_JET_MATH_UTILS_H_

#include <jet/macros.h>
#include <cstddef>
#include <limits>

namespace jet {

template <typename T>
inline T epsilon() { return std::numeric_limits<T>::epsilon(); }

template <typename T>
inline bool similar(T x, T b, T eps = epsilon<T>());

template <typename T>
inline T sign(T x);

template <typename T>
inline T min3(T x, T y, T z);

template <typename T>
inline T max3(T x, T y, T z);

//! Returns minimum among n-elements.
template <typename T>
inline T minn(const T* x, size_t n);

//! Returns maximum among n-elements.
template <typename T>
inline T maxn(const T* x, size_t n);

template <typename T>
inline T absmin(T x, T y);

template <typename T>
inline T absmax(T x, T y);

//! Returns absolute minimum among n-elements.
template <typename T>
inline T absminn(const T* x, size_t n);

//! Returns absolute maximum among n-elements.
template <typename T>
inline T absmaxn(const T* x, size_t n);

template <typename T>
inline T square(T x);

template <typename T>
inline T cubic(T x);

template <typename T>
inline T clamp(T val, T low, T high);

template <typename T>
inline T degreesToRadians(T angleInDegrees);

template <typename T>
inline T radiansToDegrees(T angleInRadians);

template<class T>
inline void getBarycentric(
    T x,
    ssize_t iLow,
    ssize_t iHigh,
    ssize_t* i,
    T* t);

template<typename S, typename T>
inline S lerp(const S& f0, const S& f1, T t);

template<typename S, typename T>
inline S bilerp(
    const S& f00, const S& f10,
    const S& f01, const S& f11,
    T tx, T ty);

template<typename S, typename T>
inline S trilerp(
    const S& f000, const S& f100,
    const S& f010, const S& f110,
    const S& f001, const S& f101,
    const S& f011, const S& f111,
    T tx, T ty, T tz);

template <typename S, typename T>
inline S catmullRom(
    const S& f0,
    const S& f1,
    const S& f2,
    const S& f3,
    T t);

template <typename T>
inline T monotonicCatmullRom(
    const T& f0,
    const T& f1,
    const T& f2,
    const T& f3,
    T t);

}  // namespace jet

#include "detail/math_utils-inl.h"

#endif  // INCLUDE_JET_MATH_UTILS_H_

