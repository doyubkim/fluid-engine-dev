// Copyright (c) 2018 Doyub Kim
//
// I am making my contributions/submissions to this project solely in my
// personal capacity and am not conveying any rights to any intellectual
// property of any third parties.

#ifndef INCLUDE_JET_DETAIL_CUDA_STD_VECTOR_INL_H_
#define INCLUDE_JET_DETAIL_CUDA_STD_VECTOR_INL_H_

#ifdef JET_USE_CUDA

#include <jet/cuda_algorithms.h>
#include <jet/cuda_std_vector.h>

#include <algorithm>

namespace jet {

template <typename T>
CudaStdVector<T>::CudaStdVector() {}

template <typename T>
CudaStdVector<T>::CudaStdVector(size_t n, const value_type& initVal) {
    resizeUninitialized(n);
    cudaFill(_ptr, n, initVal);
}

template <typename T>
template <typename A>
CudaStdVector<T>::CudaStdVector(const std::vector<T, A>& other)
    : CudaStdVector(other.size()) {
    cudaCopyHostToDevice(other.data(), _size, _ptr);
}

template <typename T>
CudaStdVector<T>::CudaStdVector(const CudaStdVector& other)
    : CudaStdVector(other.size()) {
    cudaCopyDeviceToDevice(other._ptr, _size, _ptr);
}

template <typename T>
CudaStdVector<T>::CudaStdVector(CudaStdVector&& other) {
    *this = std::move(other);
}

template <typename T>
CudaStdVector<T>::~CudaStdVector() {
    clear();
}

template <typename T>
typename CudaStdVector<T>::pointer CudaStdVector<T>::data() {
    return _ptr;
}

template <typename T>
typename CudaStdVector<T>::const_pointer CudaStdVector<T>::data() const {
    return _ptr;
}

template <typename T>
size_t CudaStdVector<T>::size() const {
    return _size;
}

#ifdef __CUDA_ARCH__
template <typename T>
__device__ typename CudaStdVector<T>::reference CudaStdVector<T>::at(size_t i) {
    return _ptr[i];
}

template <typename T>
__device__ typename CudaStdVector<T>::const_reference CudaStdVector<T>::at(
    size_t i) const {
    return _ptr[i];
}
#else
template <typename T>
typename CudaStdVector<T>::Reference CudaStdVector<T>::at(size_t i) {
    Reference r(_ptr + i);
    return r;
}

template <typename T>
T CudaStdVector<T>::at(size_t i) const {
    T tmp;
    cudaCopyDeviceToHost(_ptr + i, 1, &tmp);
    return tmp;
}
#endif

template <typename T>
void CudaStdVector<T>::clear() {
    if (_ptr != nullptr) {
        JET_CUDA_CHECK(cudaFree(_ptr));
    }
    _ptr = nullptr;
    _size = 0;
}

template <typename T>
void CudaStdVector<T>::fill(const value_type& val) {
    cudaFill(_ptr, _size, val);
}

template <typename T>
void CudaStdVector<T>::resize(size_t n, const value_type& initVal) {
    CudaStdVector newBuffer(n, initVal);
    cudaCopy(_ptr, std::min(n, _size), newBuffer._ptr);
    swap(newBuffer);
}

template <typename T>
void CudaStdVector<T>::resizeUninitialized(size_t n) {
    clear();
    JET_CUDA_CHECK(cudaMalloc(&_ptr, n * sizeof(T)));
    _size = n;
}

template <typename T>
void CudaStdVector<T>::swap(CudaStdVector& other) {
    std::swap(_ptr, other._ptr);
    std::swap(_size, other._size);
}

template <typename T>
void CudaStdVector<T>::push_back(const value_type& val) {
    CudaStdVector newBuffer;
    newBuffer.resizeUninitialized(_size + 1);
    cudaCopy(_ptr, _size, newBuffer._ptr);
    cudaCopyHostToDevice(&val, 1, newBuffer._ptr + _size);
    swap(newBuffer);
}

template <typename T>
void CudaStdVector<T>::append(const value_type& val) {
    push_back(val);
}

template <typename T>
void CudaStdVector<T>::append(const CudaStdVector& other) {
    CudaStdVector newBuffer;
    newBuffer.resizeUninitialized(_size + other._size);
    cudaCopy(_ptr, _size, newBuffer._ptr);
    cudaCopy(other._ptr, other._size, newBuffer._ptr + _size);
    swap(newBuffer);
}

template <typename T>
template <typename A>
void CudaStdVector<T>::copyFrom(const std::vector<T, A>& other) {
    if (_size == other.size()) {
        cudaCopyHostToDevice(other.data(), _size, _ptr);
    } else {
        CudaStdVector newBuffer(other);
        swap(newBuffer);
    }
}

template <typename T>
void CudaStdVector<T>::copyFrom(const CudaStdVector& other) {
    if (_size == other.size()) {
        cudaCopyDeviceToDevice(other.data(), _size, _ptr);
    } else {
        CudaStdVector newBuffer(other);
        swap(newBuffer);
    }
}

template <typename T>
template <typename A>
void CudaStdVector<T>::copyTo(std::vector<T, A>& other) {
    other.resize(_size);
    cudaCopyDeviceToHost(_ptr, _size, other.data());
}

template <typename T>
template <typename A>
CudaStdVector<T>& CudaStdVector<T>::operator=(const std::vector<T, A>& other) {
    copyFrom(other);
    return *this;
}

template <typename T>
CudaStdVector<T>& CudaStdVector<T>::operator=(const CudaStdVector& other) {
    copyFrom(other);
    return *this;
}

template <typename T>
CudaStdVector<T>& CudaStdVector<T>::operator=(CudaStdVector&& other) {
    clear();
    swap(other);
    return *this;
}

#ifdef __CUDA_ARCH__
template <typename T>
typename CudaStdVector<T>::reference CudaStdVector<T>::operator[](size_t i) {
    return at(i);
}

template <typename T>
typename CudaStdVector<T>::const_reference CudaStdVector<T>::operator[](
    size_t i) const {
    return at(i);
}
#else
template <typename T>
typename CudaStdVector<T>::Reference CudaStdVector<T>::operator[](size_t i) {
    return at(i);
}

template <typename T>
T CudaStdVector<T>::operator[](size_t i) const {
    return at(i);
}
#endif  // __CUDA_ARCH__

}  // namespace jet

#endif  // JET_USE_CUDA

#endif  // INCLUDE_JET_DETAIL_CUDA_STD_VECTOR_INL_H_
