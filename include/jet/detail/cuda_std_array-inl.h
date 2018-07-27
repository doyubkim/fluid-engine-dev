// Copyright (c) 2018 Doyub Kim
//
// I am making my contributions/submissions to this project solely in my
// personal capacity and am not conveying any rights to any intellectual
// property of any third parties.

#ifndef INCLUDE_JET_DETAIL_CUDA_STD_ARRAY_INL_H_
#define INCLUDE_JET_DETAIL_CUDA_STD_ARRAY_INL_H_

#ifdef JET_USE_CUDA

#include <jet/cuda_std_array.h>

namespace jet {

template <typename T, size_t N>
CudaStdArray<T, N>::CudaStdArray() {
    fill(T{});
}

template <typename T, size_t N>
template <typename... Args>
CudaStdArray<T, N>::CudaStdArray(const_reference first, Args... rest) {
    static_assert(
        sizeof...(Args) == N - 1,
        "Number of arguments should be equal to the size of the vector.");
    setAt(0, first, rest...);
}

template <typename T, size_t N>
CudaStdArray<T, N>::CudaStdArray(const std::array<T, N>& other) {
    for (size_t i = 0; i < N; ++i) {
        _elements[i] = other[i];
    }
}

template <typename T, size_t N>
CudaStdArray<T, N>::CudaStdArray(const Vector<T, N>& other) {
    for (size_t i = 0; i < N; ++i) {
        _elements[i] = other[i];
    }
}

template <typename T, size_t N>
CudaStdArray<T, N>::CudaStdArray(const CudaStdArray& other) {
    for (size_t i = 0; i < N; ++i) {
        _elements[i] = other[i];
    }
}

template <typename T, size_t N>
void CudaStdArray<T, N>::fill(const_reference val) {
    for (size_t i = 0; i < N; ++i) {
        _elements[i] = val;
    }
}

template <typename T, size_t N>
typename CudaStdArray<T, N>::reference CudaStdArray<T, N>::operator[](
    size_t i) {
    return _elements[i];
}

template <typename T, size_t N>
typename CudaStdArray<T, N>::const_reference CudaStdArray<T, N>::operator[](
    size_t i) const {
    return _elements[i];
}

template <typename T, size_t N>
bool CudaStdArray<T, N>::operator==(const CudaStdArray& other) const {
    for (size_t i = 0; i < N; ++i) {
        if (_elements[i] != other._elements[i]) {
            return false;
        }
    }

    return true;
}

template <typename T, size_t N>
bool CudaStdArray<T, N>::operator!=(const CudaStdArray& other) const {
    return !(*this == other);
}

template <typename T, size_t N>
template <typename... Args>
void CudaStdArray<T, N>::setAt(size_t i, const_reference first, Args... rest) {
    _elements[i] = first;
    setAt(i + 1, rest...);
}

template <typename T, size_t N>
template <typename... Args>
void CudaStdArray<T, N>::setAt(size_t i, const_reference first) {
    _elements[i] = first;
}

}  // namespace jet

#endif  // JET_USE_CUDA

#endif  // INCLUDE_JET_DETAIL_CUDA_STD_ARRAY_INL_H_
