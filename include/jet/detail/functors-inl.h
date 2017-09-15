// Copyright (c) 2017 Doyub Kim
//
// I am making my contributions/submissions to this project solely in my
// personal capacity and am not conveying any rights to any intellectual
// property of any third parties.

#ifndef INCLUDE_JET_DETAIL_FUNCTORS_INL_H_
#define INCLUDE_JET_DETAIL_FUNCTORS_INL_H_

#include <jet/functors.h>

namespace jet {

template <typename T, typename U>
constexpr U TypeCast<T, U>::operator()(const T& a) const {
    return static_cast<U>(a);
}

template <typename T>
constexpr T RMinus<T>::operator()(const T& a, const T& b) const {
    return b - a;
}

template <typename T>
constexpr T RDivides<T>::operator()(const T& a, const T& b) const {
    return b / a;
}
}

#endif  // INCLUDE_JET_DETAIL_FUNCTORS_INL_H_
