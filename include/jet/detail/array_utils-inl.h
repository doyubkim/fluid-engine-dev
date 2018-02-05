// Copyright (c) 2018 Doyub Kim
//
// I am making my contributions/submissions to this project solely in my
// personal capacity and am not conveying any rights to any intellectual
// property of any third parties.

#ifndef INCLUDE_JET_DETAIL_ARRAY_UTILS_INL_H_
#define INCLUDE_JET_DETAIL_ARRAY_UTILS_INL_H_

#include <jet/array1.h>
#include <jet/array2.h>
#include <jet/array3.h>
#include <jet/parallel.h>
#include <jet/serial.h>
#include <jet/type_helpers.h>
#include <iostream>

namespace jet {

template <typename ArrayType, typename T>
void setRange1(
    size_t size,
    const T& value,
    ArrayType* output) {
    setRange1(kZeroSize, size, value, output);
}

template <typename ArrayType, typename T>
void setRange1(
    size_t begin,
    size_t end,
    const T& value,
    ArrayType* output) {
    parallelFor(
        begin,
        end,
        [&](size_t i) {
            (*output)[i] = value;
        });
}

template <typename ArrayType1, typename ArrayType2>
void copyRange1(
    const ArrayType1& input,
    size_t size,
    ArrayType2* output) {
    copyRange1(input, 0, size, output);
}

template <typename ArrayType1, typename ArrayType2>
void copyRange1(
    const ArrayType1& input,
    size_t begin,
    size_t end,
    ArrayType2* output) {
    parallelFor(begin, end,
        [&input, &output](size_t i) {
            (*output)[i] = input[i];
        });
}

template <typename ArrayType1, typename ArrayType2>
void copyRange2(
    const ArrayType1& input,
    size_t sizeX,
    size_t sizeY,
    ArrayType2* output) {
    copyRange2(input, kZeroSize, sizeX, kZeroSize, sizeY, output);
}

template <typename ArrayType1, typename ArrayType2>
void copyRange2(
    const ArrayType1& input,
    size_t beginX,
    size_t endX,
    size_t beginY,
    size_t endY,
    ArrayType2* output) {
    parallelFor(beginX, endX, beginY, endY,
        [&input, &output](size_t i, size_t j) {
            (*output)(i, j) = input(i, j);
        });
}

template <typename ArrayType1, typename ArrayType2>
void copyRange3(
    const ArrayType1& input,
    size_t sizeX,
    size_t sizeY,
    size_t sizeZ,
    ArrayType2* output) {
    copyRange3(
        input, kZeroSize, sizeX, kZeroSize, sizeY, kZeroSize, sizeZ, output);
}

template <typename ArrayType1, typename ArrayType2>
void copyRange3(
    const ArrayType1& input,
    size_t beginX,
    size_t endX,
    size_t beginY,
    size_t endY,
    size_t beginZ,
    size_t endZ,
    ArrayType2* output) {
    parallelFor(beginX, endX, beginY, endY, beginZ, endZ,
        [&input, &output](size_t i, size_t j, size_t k) {
            (*output)(i, j, k) = input(i, j, k);
        });
}

template <typename T>
void extrapolateToRegion(
    const ConstArrayAccessor2<T>& input,
    const ConstArrayAccessor2<char>& valid,
    unsigned int numberOfIterations,
    ArrayAccessor2<T> output) {
    const Size2 size = input.size();

    JET_ASSERT(size == valid.size());
    JET_ASSERT(size == output.size());

    Array2<char> valid0(size);
    Array2<char> valid1(size);

    valid0.parallelForEachIndex([&](size_t i, size_t j) {
        valid0(i, j) = valid(i, j);
        output(i, j) = input(i, j);
    });

    for (unsigned int iter = 0; iter < numberOfIterations; ++iter) {
        valid0.forEachIndex([&](size_t i, size_t j) {
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
                    output(i, j)
                        = sum
                        / static_cast<typename ScalarType<T>::value>(count);
                    valid1(i, j) = 1;
                }
            } else {
                valid1(i, j) = 1;
            }
        });

        valid0.swap(valid1);
    }
}

template <typename T>
void extrapolateToRegion(
    const ConstArrayAccessor3<T>& input,
    const ConstArrayAccessor3<char>& valid,
    unsigned int numberOfIterations,
    ArrayAccessor3<T> output) {
    const Size3 size = input.size();

    JET_ASSERT(size == valid.size());
    JET_ASSERT(size == output.size());

    Array3<char> valid0(size);
    Array3<char> valid1(size);

    valid0.parallelForEachIndex([&](size_t i, size_t j, size_t k) {
        valid0(i, j, k) = valid(i, j, k);
        output(i, j, k) = input(i, j, k);
    });

    for (unsigned int iter = 0; iter < numberOfIterations; ++iter) {
        valid0.forEachIndex([&](size_t i, size_t j, size_t k) {
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
                    output(i, j, k)
                        = sum
                        / static_cast<typename ScalarType<T>::value>(count);
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
