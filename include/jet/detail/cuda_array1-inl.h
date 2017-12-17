// Copyright (c) 2017 Doyub Kim
//
// I am making my contributions/submissions to this project solely in my
// personal capacity and am not conveying any rights to any intellectual
// property of any third parties.

#ifdef JET_USE_CUDA

#ifndef INCLUDE_JET_DETAIL_CUDA_ARRAY1_INL_H_
#define INCLUDE_JET_DETAIL_CUDA_ARRAY1_INL_H_

#include <jet/cuda_array1.h>

namespace jet {

namespace experimental {

template <typename T>
CudaArray1<T>::CudaArray1() {}

template <typename T>
CudaArray1<T>::CudaArray1(size_t size, const T& initVal) {
    resize(size, initVal);
}

template <typename T>
CudaArray1<T>::CudaArray1(const CudaArray1& other) {
    set(other);
}

template <typename T>
void CudaArray1<T>::set(const T& value) {
    thrust::fill(_data.begin(), _data.end(), value);
}

template <typename T>
void CudaArray1<T>::set(const CudaArray1& other) {
    _data = other._data;
}

template <typename T>
void CudaArray1<T>::clear() {
    _data.clear();
}

template <typename T>
void CudaArray1<T>::resize(size_t size, const T& initVal) {
    _data.resize(size, initVal);
}

template <typename T>
size_t CudaArray1<T>::size() const {
    return _data.size();
}

template <typename T>
typename CudaArray1<T>::ContainerType::reference CudaArray1<T>::operator[](size_t i) {
    return _data[i];
}

template <typename T>
const T& CudaArray1<T>::operator[](size_t i) const {
    return _data[i];
}

}  // namespace experimental

}  // namespace jet

#endif  // INCLUDE_JET_DETAIL_CUDA_ARRAY1_INL_H_

#endif  // JET_USE_CUDA
