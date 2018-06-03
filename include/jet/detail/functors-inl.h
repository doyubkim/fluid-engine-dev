// Copyright (c) 2018 Doyub Kim
//
// I am making my contributions/submissions to this project solely in my
// personal capacity and am not conveying any rights to any intellectual
// property of any third parties.

#ifndef INCLUDE_JET_DETAIL_FUNCTORS_INL_H_
#define INCLUDE_JET_DETAIL_FUNCTORS_INL_H_

#include <jet/functors.h>
#include <jet/math_utils.h>

namespace jet {

template <typename T, typename U>
constexpr U TypeCast<T, U>::operator()(const T& a) const {
    return static_cast<U>(a);
}

template <typename T>
constexpr T Ceil<T>::operator()(const T& a) const {
    return std::ceil(a);
}

template <typename T>
constexpr T Floor<T>::operator()(const T& a) const {
    return std::floor(a);
}

template <typename T>
constexpr T RMinus<T>::operator()(const T& a, const T& b) const {
    return b - a;
}

template <typename T>
constexpr T RDivides<T>::operator()(const T& a, const T& b) const {
    return b / a;
}

template <typename T>
void IAdd<T>::operator()(T& a, const T& b) const {
    a += b;
}

template <typename T>
void ISub<T>::operator()(T& a, const T& b) const {
    a -= b;
}

template <typename T>
void IMul<T>::operator()(T& a, const T& b) const {
    a *= b;
}

template <typename T>
void IDiv<T>::operator()(T& a, const T& b) const {
    a /= b;
}

template <typename T>
constexpr T Min<T>::operator()(const T& a, const T& b) const {
    return std::min(a, b);
}

template <typename T>
constexpr T Max<T>::operator()(const T& a, const T& b) const {
    return std::max(a, b);
}

template <typename T>
constexpr T Clamp<T>::operator()(const T& a, const T& low,
                                 const T& high) const {
    return clamp(a, low, high);
}

}  // namespace jet

#endif  // INCLUDE_JET_DETAIL_FUNCTORS_INL_H_
