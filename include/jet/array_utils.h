// Copyright (c) 2018 Doyub Kim
//
// I am making my contributions/submissions to this project solely in my
// personal capacity and am not conveying any rights to any intellectual
// property of any third parties.

#ifndef INCLUDE_JET_ARRAY_UTILS_H_
#define INCLUDE_JET_ARRAY_UTILS_H_

#include <jet/array.h>
#include <jet/array_view.h>

namespace jet {

template <typename T, size_t N>
void fill(ArrayView<T, N> a, const Vector<size_t, N>& begin,
          const Vector<size_t, N>& end, const T& val);

template <typename T, size_t N>
void fill(ArrayView<T, N> a, const T& val);

template <typename T>
void fill(ArrayView<T, 1> a, size_t begin, size_t end, const T& val);

template <typename T, typename U, size_t N>
void copy(ArrayView<T, N> src, const Vector<size_t, N>& begin,
          const Vector<size_t, N>& end, ArrayView<U, N> dst);

template <typename T, typename U, size_t N>
void copy(ArrayView<T, N> src, ArrayView<U, N> dst);

template <typename T, typename U>
void copy(ArrayView<T, 1> src, size_t begin, size_t end, ArrayView<U, 1> dst);

//!
//! \brief Extrapolates 2-D input data from 'valid' (1) to 'invalid' (0) region.
//!
//! This function extrapolates 2-D input data from 'valid' (1) to 'invalid' (0)
//! region. It iterates multiple times to propagate the 'valid' values to nearby
//! 'invalid' region. The maximum distance of the propagation is equal to
//! numberOfIterations. The input parameters 'valid' and 'data' should be
//! collocated.
//!
//! \param input - data to extrapolate
//! \param valid - set 1 if valid, else 0.
//! \param numberOfIterations - number of iterations for propagation
//! \param output - extrapolated output
//!
template <typename T, typename U>
void extrapolateToRegion(ArrayView2<T> input, ArrayView2<char> valid,
                         unsigned int numberOfIterations, ArrayView2<U> output);

//!
//! \brief Extrapolates 3-D input data from 'valid' (1) to 'invalid' (0) region.
//!
//! This function extrapolates 3-D input data from 'valid' (1) to 'invalid' (0)
//! region. It iterates multiple times to propagate the 'valid' values to nearby
//! 'invalid' region. The maximum distance of the propagation is equal to
//! numberOfIterations. The input parameters 'valid' and 'data' should be
//! collocated.
//!
//! \param input - data to extrapolate
//! \param valid - set 1 if valid, else 0.
//! \param numberOfIterations - number of iterations for propagation
//! \param output - extrapolated output
//!
template <typename T, typename U>
void extrapolateToRegion(ArrayView3<T> input, ArrayView3<char> valid,
                         unsigned int numberOfIterations, ArrayView3<U> output);

}  // namespace jet

#include <jet/detail/array_utils-inl.h>

#endif  // INCLUDE_JET_ARRAY_UTILS_H_
