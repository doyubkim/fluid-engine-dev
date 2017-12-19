// Copyright (c) 2017 Doyub Kim
//
// I am making my contributions/submissions to this project solely in my
// personal capacity and am not conveying any rights to any intellectual
// property of any third parties.

#ifdef JET_USE_CUDA

#ifndef INCLUDE_JET_DETAIL_CUDA_ARRAY_VIEW1_INL_H_
#define INCLUDE_JET_DETAIL_CUDA_ARRAY_VIEW1_INL_H_

#include <jet/cuda_array_view1.h>

#include <thrust/device_ptr.h>

namespace jet {

namespace experimental {

template <typename T>
CudaArrayView1<T>::CudaArrayView1() {}

template <typename T>
CudaArrayView1<T>::CudaArrayView1(T* data, size_t size) {
    set(data, size);
}

template <typename T>
CudaArrayView1<T>::CudaArrayView1(const CudaArray1<T>& array) {
    set(array);
}

template <typename T>
CudaArrayView1<T>::CudaArrayView1(const thrust::device_vector<T>& vec) {
    set(vec);
}

template <typename T>
CudaArrayView1<T>::CudaArrayView1(const CudaArrayView1& other) {
    set(other);
}

template <typename T>
void CudaArrayView1<T>::set(const T& value) {
    thrust::fill(_data, _data + _size, value);
}

template <typename T>
void CudaArrayView1<T>::set(T* data, size_t size) {
    _data = thrust::device_pointer_cast(data);
    _size = size;
}

template <typename T>
void CudaArrayView1<T>::set(const CudaArray1<T>& array) {
    set(array.view());
}

template <typename T>
void CudaArrayView1<T>::set(const thrust::device_vector<T>& vec) {
    _data = vec.data();
    _size = vec.size();
}

template <typename T>
void CudaArrayView1<T>::set(const CudaArrayView1& other) {
    _data = other._data;
    _size = other._size;
}

template <typename T>
size_t CudaArrayView1<T>::size() const {
    return _size;
}

template <typename T>
T* CudaArrayView1<T>::data() {
    return thrust::raw_pointer_cast(_data);
}

template <typename T>
const T* CudaArrayView1<T>::data() const {
    return thrust::raw_pointer_cast(_data);
}

template <typename T>
thrust::device_ptr<T> CudaArrayView1<T>::begin() const {
    return _data;
}

template <typename T>
thrust::device_ptr<T> CudaArrayView1<T>::end() const {
    return _data + _size;
}

template <typename T>
typename thrust::device_ptr<T>::reference CudaArrayView1<T>::operator[](
    size_t i) {
    return _data[i];
}

template <typename T>
const T& CudaArrayView1<T>::operator[](size_t i) const {
    return _data[i];
}

}  // namespace experimental

}  // namespace jet

#endif  // INCLUDE_JET_DETAIL_CUDA_ARRAY_VIEW1_INL_H_

#endif  // JET_USE_CUDA
