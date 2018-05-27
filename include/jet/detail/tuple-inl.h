// Copyright (c) 2018 Doyub Kim
//
// I am making my contributions/submissions to this project solely in my
// personal capacity and am not conveying any rights to any intellectual
// property of any third parties.

#ifndef INCLUDE_JET_DETAIL_TUPLE_INL_H_
#define INCLUDE_JET_DETAIL_TUPLE_INL_H_

#include <jet/tuple.h>

namespace jet {

template <typename T, size_t N>
T& Tuple<T, N>::operator[](size_t i) {
    JET_ASSERT(i < N);
    return _elements[i];
}

template <typename T, size_t N>
const T& Tuple<T, N>::operator[](size_t i) const {
    JET_ASSERT(i < N);
    return _elements[i];
}

//

template <typename T>
T& Tuple<T, 1>::operator[](size_t i) {
    JET_ASSERT(i < 1);
    return (&x)[i];
}

template <typename T>
const T& Tuple<T, 1>::operator[](size_t i) const {
    JET_ASSERT(i < 1);
    return (&x)[i];
}

//

template <typename T>
T& Tuple<T, 2>::operator[](size_t i) {
    JET_ASSERT(i < 2);
    return (&x)[i];
}

template <typename T>
const T& Tuple<T, 2>::operator[](size_t i) const {
    JET_ASSERT(i < 2);
    return (&x)[i];
}

//

template <typename T>
T& Tuple<T, 3>::operator[](size_t i) {
    JET_ASSERT(i < 3);
    return (&x)[i];
}

template <typename T>
const T& Tuple<T, 3>::operator[](size_t i) const {
    JET_ASSERT(i < 3);
    return (&x)[i];
}

//

template <typename T>
T& Tuple<T, 4>::operator[](size_t i) {
    JET_ASSERT(i < 4);
    return (&x)[i];
}

template <typename T>
const T& Tuple<T, 4>::operator[](size_t i) const {
    JET_ASSERT(i < 4);
    return (&x)[i];
}

}  // namespace jet

#endif  // INCLUDE_JET_DETAIL_TUPLE_INL_H_
