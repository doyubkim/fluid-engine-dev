// Copyright (c) 2018 Doyub Kim
//
// I am making my contributions/submissions to this project solely in my
// personal capacity and am not conveying any rights to any intellectual
// property of any third parties.

#ifndef INCLUDE_JET_MATH_UTILS_H_
#define INCLUDE_JET_MATH_UTILS_H_

#include <jet/macros.h>
#include <cstddef>
#include <limits>

namespace jet {

//!
//! \brief      Returns true if \p x and \p y are similar.
//!
//! \param[in]  x     The first value.
//! \param[in]  y     The second value.
//! \param[in]  eps   The tolerance.
//!
//! \tparam     T     Value type.
//!
//! \return     True if similar.
//!
template <typename T>
inline bool similar(T x, T y, T eps = std::numeric_limits<T>::epsilon());

//!
//! \brief      Returns the sign of the value.
//!
//! \param[in]  x     Input value.
//!
//! \tparam     T     Value type.
//!
//! \return     The sign.
//!
template <typename T>
inline T sign(T x);

//!
//! \brief      Returns the minimum value among three inputs.
//!
//! \param[in]  x     The first value.
//! \param[in]  y     The second value.
//! \param[in]  z     The three value.
//!
//! \tparam     T     Value type.
//!
//! \return     The minimum value.
//!
template <typename T>
inline T min3(T x, T y, T z);

//!
//! \brief      Returns the maximum value among three inputs.
//!
//! \param[in]  x     The first value.
//! \param[in]  y     The second value.
//! \param[in]  z     The three value.
//!
//! \tparam     T     Value type.
//!
//! \return     The maximum value.
//!
template <typename T>
inline T max3(T x, T y, T z);

//! Returns minimum among n-elements.
template <typename T>
inline T minn(const T* x, size_t n);

//! Returns maximum among n-elements.
template <typename T>
inline T maxn(const T* x, size_t n);

//!
//! \brief      Returns the absolute minimum value among the two inputs.
//!
//! \param[in]  x     The first value.
//! \param[in]  y     The second value.
//!
//! \tparam     T     Value type.
//!
//! \return     The absolute minimum.
//!
template <typename T>
inline T absmin(T x, T y);

//!
//! \brief      Returns the absolute maximum value among the two inputs.
//!
//! \param[in]  x     The first value.
//! \param[in]  y     The second value.
//!
//! \tparam     T     Value type.
//!
//! \return     The absolute maximum.
//!
template <typename T>
inline T absmax(T x, T y);

//! Returns absolute minimum among n-elements.
template <typename T>
inline T absminn(const T* x, size_t n);

//! Returns absolute maximum among n-elements.
template <typename T>
inline T absmaxn(const T* x, size_t n);

template <typename T>
inline size_t argmin2(T x, T y);

template <typename T>
inline size_t argmax2(T x, T y);

template <typename T>
inline size_t argmin3(T x, T y, T z);

template <typename T>
inline size_t argmax3(T x, T y, T z);

//!
//! \brief      Returns the square of \p x.
//!
//! \param[in]  x     The input.
//!
//! \tparam     T     Value type.
//!
//! \return     The squared value.
//!
template <typename T>
inline T square(T x);

//!
//! \brief      Returns the cubic of \p x.
//!
//! \param[in]  x     The input.
//!
//! \tparam     T     Value type.
//!
//! \return     The cubic of \p x.
//!
template <typename T>
inline T cubic(T x);

//!
//! \brief      Returns the clamped value.
//!
//! \param[in]  val   The value.
//! \param[in]  low   The low value.
//! \param[in]  high  The high value.
//!
//! \tparam     T     Value type.
//!
//! \return     The clamped value.
//!
template <typename T>
inline T clamp(T val, T low, T high);

//!
//! \brief      Converts degrees to radians.
//!
//! \param[in]  angleInDegrees The angle in degrees.
//!
//! \tparam     T              Value type.
//!
//! \return     Angle in radians.
//!
template <typename T>
inline T degreesToRadians(T angleInDegrees);

//!
//! \brief      Converts radians to degrees.
//!
//! \param[in]  angleInDegrees The angle in radians.
//!
//! \tparam     T              Value type.
//!
//! \return     Angle in degrees.
//!
template <typename T>
inline T radiansToDegrees(T angleInRadians);

//!
//! \brief      Gets the barycentric coordinate.
//!
//! \param[in]  x     The input value.
//! \param[in]  iLow  The lowest index.
//! \param[in]  iHigh The highest index.
//! \param      i     The output index.
//! \param      t     The offset from \p i.
//!
//! \tparam     T     Value type.
//!
template <class T>
inline void getBarycentric(T x, ssize_t iLow, ssize_t iHigh, ssize_t* i, T* t);

//!
//! \brief      Computes linear interpolation.
//!
//! \param[in]  f0    The first value.
//! \param[in]  f1    The second value.
//! \param[in]  t     Relative offset [0, 1] from the first value.
//!
//! \tparam     S     Input value type.
//! \tparam     T     Offset type.
//!
//! \return     The interpolated value.
//!
template <typename S, typename T>
inline S lerp(const S& f0, const S& f1, T t);

//! \brief      Computes bilinear interpolation.
template <typename S, typename T>
inline S bilerp(const S& f00, const S& f10, const S& f01, const S& f11, T tx,
                T ty);

//! \brief      Computes trilinear interpolation.
template <typename S, typename T>
inline S trilerp(const S& f000, const S& f100, const S& f010, const S& f110,
                 const S& f001, const S& f101, const S& f011, const S& f111,
                 T tx, T ty, T tz);

//! \brief      Computes Catmull-Rom interpolation.
template <typename S, typename T>
inline S catmullRom(const S& f0, const S& f1, const S& f2, const S& f3, T t);

//! \brief      Computes monotonic Catmull-Rom interpolation.
template <typename T>
inline T monotonicCatmullRom(const T& f0, const T& f1, const T& f2, const T& f3,
                             T t);

}  // namespace jet

#include "detail/math_utils-inl.h"

#endif  // INCLUDE_JET_MATH_UTILS_H_
