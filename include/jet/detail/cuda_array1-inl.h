// Copyright (c) 2017 Doyub Kim
//
// I am making my contributions/submissions to this project solely in my
// personal capacity and am not conveying any rights to any intellectual
// property of any third parties.

#ifdef JET_USE_CUDA

#ifndef INCLUDE_JET_DETAIL_CUDA_ARRAY1_INL_H_
#define INCLUDE_JET_DETAIL_CUDA_ARRAY1_INL_H_

#include <jet/cuda_array1.h>
#include <jet/cuda_array_view1.h>

namespace jet {

namespace experimental {

template <typename T>
CudaArray1<T>::CudaArray1() {}

template <typename T>
CudaArray1<T>::CudaArray1(size_t size, const T& initVal) {
    resize(size, initVal);
}

template <typename T>
CudaArray1<T>::CudaArray1(const CudaArrayView1<T>& view) {
    set(view);
}

template <typename T>
CudaArray1<T>::CudaArray1(const thrust::device_vector<T>& vec) {
    set(vec);
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
void CudaArray1<T>::set(const CudaArrayView1<T>& view) {
    resize(view.size());
    auto v = thrust::device_pointer_cast(view.data());
    thrust::copy(_data.begin(), _data.end(), v, v + view.size());
}

template <typename T>
void CudaArray1<T>::set(const thrust::device_vector<T>& vec) {
    _data = vec;
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
T* CudaArray1<T>::data() {
    return thrust::raw_pointer_cast(_data.data());
}

template <typename T>
const T* CudaArray1<T>::data() const {
    return thrust::raw_pointer_cast(_data.data());
}

template <typename T>
CudaArrayView1<T> CudaArray1<T>::view() {
    return CudaArrayView1<T>(data(), size());
}

template <typename T>
const CudaArrayView1<T> CudaArray1<T>::view() const {
    return CudaArrayView1<T>(thrust::raw_pointer_cast(_data.data()), size());
}

template <typename T>
typename CudaArray1<T>::ContainerType::reference CudaArray1<T>::operator[](
    size_t i) {
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
