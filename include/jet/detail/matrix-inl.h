// Copyright (c) 2017 Doyub Kim
//
// I am making my contributions/submissions to this project solely in my
// personal capacity and am not conveying any rights to any intellectual
// property of any third parties.

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
