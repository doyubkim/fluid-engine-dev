// Copyright (c) 2018 Doyub Kim
//
// I am making my contributions/submissions to this project solely in my
// personal capacity and am not conveying any rights to any intellectual
// property of any third parties.

#ifndef INCLUDE_JET_DETAIL_ARRAY_UTILS_INL_H_
#define INCLUDE_JET_DETAIL_ARRAY_UTILS_INL_H_

#include <jet/array.h>
#include <jet/iteration_utils.h>
#include <jet/type_helpers.h>

namespace jet {

template <typename T, size_t N>
void fill(ArrayView<T, N> a, const SizeN<N>& begin, const SizeN<N>& end,
          const T& val) {
    forEachIndex(begin, end, [&](auto... idx) { a(idx...) = val; });
}

template <typename T, size_t N>
void fill(ArrayView<T, N> a, const T& val) {
    fill(a, SizeN<N>{}, SizeN<N>{a.size()}, val);
}

template <typename T>
void fill(ArrayView<T, 1> a, size_t begin, size_t end, const T& val) {
    fill(a, Size1{begin}, Size1{end}, val);
}

template <typename T, typename U, size_t N>
void copy(ArrayView<T, N> src, const SizeN<N>& begin, const SizeN<N>& end,
          ArrayView<U, N> dst) {
    forEachIndex(begin, end, [&](auto... idx) { dst(idx...) = src(idx...); });
}

template <typename T, typename U, size_t N>
void copy(ArrayView<T, N> src, ArrayView<U, N> dst) {
    copy(src, SizeN<N>{}, SizeN<N>{src.size()}, dst);
}

template <typename T, typename U>
void copy(ArrayView<T, 1> src, size_t begin, size_t end, ArrayView<U, 1> dst) {
    copy(src, Size1{begin}, Size1{end}, dst);
}

template <typename T, typename U>
void extrapolateToRegion(const ArrayView2<T>& input,
                         const ConstArrayView2<char>& valid,
                         unsigned int numberOfIterations,
                         ArrayView2<U> output) {
    const Size2 size = input.size();

    JET_ASSERT(size == valid.size());
    JET_ASSERT(size == output.size());

    Array2<char> valid0(size);
    Array2<char> valid1(size);

    parallelForEachIndex(valid0.size(), [&](size_t i, size_t j) {
        valid0(i, j) = valid(i, j);
        output(i, j) = input(i, j);
    });

    for (unsigned int iter = 0; iter < numberOfIterations; ++iter) {
        forEachIndex(valid0.size(), [&](size_t i, size_t j) {
            T sum = zero<T>();
            unsigned int count = 0;

            if (!valid0(i, j)) {
                if (i + 1 < size.x && valid0(i + 1, j)) {
                    sum += output(i + 1, j);
                    ++count;
                }

                if (i > 0 && valid0(i - 1, j)) {
                    sum += output(i - 1, j);
                    ++count;
                }

                if (j + 1 < size.y && valid0(i, j + 1)) {
                    sum += output(i, j + 1);
                    ++count;
                }

                if (j > 0 && valid0(i, j - 1)) {
                    sum += output(i, j - 1);
                    ++count;
                }

                if (count > 0) {
                    output(i, j) =
                        sum / static_cast<typename ScalarType<T>::value>(count);
                    valid1(i, j) = 1;
                }
            } else {
                valid1(i, j) = 1;
            }
        });

        valid0.swap(valid1);
    }
}

template <typename T, typename U>
void extrapolateToRegion(const ArrayView3<T>& input,
                         const ConstArrayView3<char>& valid,
                         unsigned int numberOfIterations,
                         ArrayView3<U> output) {
    const Size3 size = input.size();

    JET_ASSERT(size == valid.size());
    JET_ASSERT(size == output.size());

    Array3<char> valid0(size);
    Array3<char> valid1(size);

    parallelForEachIndex(valid0.size(), [&](size_t i, size_t j, size_t k) {
        valid0(i, j, k) = valid(i, j, k);
        output(i, j, k) = input(i, j, k);
    });

    for (unsigned int iter = 0; iter < numberOfIterations; ++iter) {
        forEachIndex(valid0.size(), [&](size_t i, size_t j, size_t k) {
            T sum = zero<T>();
            unsigned int count = 0;

            if (!valid0(i, j, k)) {
                if (i + 1 < size.x && valid0(i + 1, j, k)) {
                    sum += output(i + 1, j, k);
                    ++count;
                }

                if (i > 0 && valid0(i - 1, j, k)) {
                    sum += output(i - 1, j, k);
                    ++count;
                }

                if (j + 1 < size.y && valid0(i, j + 1, k)) {
                    sum += output(i, j + 1, k);
                    ++count;
                }

                if (j > 0 && valid0(i, j - 1, k)) {
                    sum += output(i, j - 1, k);
                    ++count;
                }

                if (k + 1 < size.z && valid0(i, j, k + 1)) {
                    sum += output(i, j, k + 1);
                    ++count;
                }

                if (k > 0 && valid0(i, j, k - 1)) {
                    sum += output(i, j, k - 1);
                    ++count;
                }

                if (count > 0) {
                    output(i, j, k) =
                        sum / static_cast<typename ScalarType<T>::value>(count);
                    valid1(i, j, k) = 1;
                }
            } else {
                valid1(i, j, k) = 1;
            }
        });

        valid0.swap(valid1);
    }
}

}  // namespace jet

#endif  // INCLUDE_JET_DETAIL_ARRAY_UTILS_INL_H_
