// Copyright (c) 2016 Doyub Kim

#ifndef INCLUDE_JET_MATRIX_H_
#define INCLUDE_JET_MATRIX_H_

#include <jet/macros.h>
#include <array>
#include <type_traits>

namespace jet {

//!
//! \brief Static-sized M x N matrix class.
//!
//! TODO(issue #3) : Implement statically-sized MxN matrix implementation
//!
//! \tparam T - Real number type.
//! \tparam M - Number of rows.
//! \tparam N - Number of columns.
//!
template <typename T, size_t M, size_t N>
class Matrix {
 public:
    static_assert(
        M > 0,
        "Number of rows for static-sized matrix should be greater than zero.");
    static_assert(
        N > 0,
        "Number of columns for static-sized matrix should be greater than "
        "zero.");
    static_assert(
        std::is_floating_point<T>::value,
        "Matrix only can be instantiated with floating point types");

    std::array<T, M * N> elements;

    //! Default constructor.
    //! \warning This constructor will create zero matrix.
    Matrix();
};

}  // namespace jet

#include "detail/matrix-inl.h"

#endif  // INCLUDE_JET_MATRIX_H_
