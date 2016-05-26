// Copyright (c) 2016 Doyub Kim

#ifndef INCLUDE_JET_DETAIL_MATRIX_INL_H_
#define INCLUDE_JET_DETAIL_MATRIX_INL_H_

#include <cassert>

namespace jet {

template <typename T, size_t M, size_t N>
Matrix<T, M, N>::Matrix() {
    for (auto& elem : elements) {
        elem = 0;
    }
}

}  // namespace jet

#endif  // INCLUDE_JET_DETAIL_MATRIX_INL_H_
