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
CudaArray1<T>::CudaArray1(const ArrayView1<T>& view) {
    set(view);
}

template <typename T>
CudaArray1<T>::CudaArray1(const CudaArrayView1<T>& view) {
    set(view);
}

template <typename T>
CudaArray1<T>::CudaArray1(const std::initializer_list<T>& lst) {
    set(lst);
}

template <typename T>
template <typename Alloc>
CudaArray1<T>::CudaArray1(const std::vector<T, Alloc>& vec) {
    set(vec);
}

template <typename T>
template <typename Alloc>
CudaArray1<T>::CudaArray1(const thrust::host_vector<T, Alloc>& vec) {
    set(vec);
}

template <typename T>
template <typename Alloc>
CudaArray1<T>::CudaArray1(const thrust::device_vector<T, Alloc>& vec) {
    set(vec);
}

template <typename T>
CudaArray1<T>::CudaArray1(const CudaArray1& other) {
    set(other);
}

template <typename T>
CudaArray1<T>::CudaArray1(CudaArray1&& other) {
    (*this) = std::move(other);
}

template <typename T>
void CudaArray1<T>::set(const T& value) {
    thrust::fill(_data.begin(), _data.end(), value);
}

template <typename T>
void CudaArray1<T>::set(const ArrayView1<T>& view) {
    size_t n = view.size();
    thrust::host_vector<T> temp(n);
    for (size_t i = 0; i < n; ++i) {
        temp[i] = view[i];
    }
    set(temp);
}

template <typename T>
void CudaArray1<T>::set(const CudaArrayView1<T>& view) {
    resize(view.size());
    thrust::copy(view.data(), view.data() + view.size(), _data.begin());
}

template <typename T>
void CudaArray1<T>::set(const std::initializer_list<T>& lst) {
    thrust::host_vector<T> temp;
    for (const auto& v : lst) {
        temp.push_back(v);
    }
    _data = temp;
}

template <typename T>
template <typename Alloc>
void CudaArray1<T>::set(const std::vector<T, Alloc>& vec) {
    // Workaround for warning : calling a __host__ function... from a __host__
    // __device__ function ...
    size_t n = vec.size();
    thrust::host_vector<T> temp(n);
    for (size_t i = 0; i < n; ++i) {
        temp[i] = vec[i];
    }
    set(temp);
}

template <typename T>
template <typename Alloc>
void CudaArray1<T>::set(const thrust::host_vector<T, Alloc>& vec) {
    _data = vec;
}

template <typename T>
template <typename Alloc>
void CudaArray1<T>::set(const thrust::device_vector<T, Alloc>& vec) {
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
void CudaArray1<T>::swap(CudaArray1& other) {
    _data.swap(other._data);
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
typename CudaArray1<T>::Iterator CudaArray1<T>::begin() {
    return _data.data();
}

template <typename T>
typename CudaArray1<T>::Iterator CudaArray1<T>::begin() const {
    return _data.data();
}

template <typename T>
typename CudaArray1<T>::Iterator CudaArray1<T>::end() {
    return _data.data() + _data.size();
}

template <typename T>
typename CudaArray1<T>::Iterator CudaArray1<T>::end() const {
    return _data.data() + _data.size();
}

template <typename T>
CudaArrayView1<T> CudaArray1<T>::view() {
    return CudaArrayView1<T>(*this);
}

template <typename T>
const CudaArrayView1<T> CudaArray1<T>::view() const {
    return CudaArrayView1<T>(*this);
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

template <typename T>
CudaArray1<T>& CudaArray1<T>::operator=(const T& value) {
    set(value);
    return *this;
}

template <typename T>
CudaArray1<T>& CudaArray1<T>::operator=(const ArrayView1<T>& view) {
    set(view);
    return *this;
}

template <typename T>
CudaArray1<T>& CudaArray1<T>::operator=(const CudaArrayView1<T>& view) {
    set(view);
    return *this;
}

template <typename T>
template <typename Alloc>
CudaArray1<T>& CudaArray1<T>::operator=(const std::vector<T, Alloc>& vec) {
    set(vec);
    return *this;
}

template <typename T>
template <typename Alloc>
CudaArray1<T>& CudaArray1<T>::operator=(
    const thrust::host_vector<T, Alloc>& vec) {
    set(vec);
    return *this;
}

template <typename T>
template <typename Alloc>
CudaArray1<T>& CudaArray1<T>::operator=(
    const thrust::device_vector<T, Alloc>& vec) {
    set(vec);
    return *this;
}

template <typename T>
CudaArray1<T>& CudaArray1<T>::operator=(const CudaArray1& other) {
    set(other);
    return *this;
}

template <typename T>
CudaArray1<T>& CudaArray1<T>::operator=(CudaArray1&& other) {
    swap(other);
    other.clear();
    return *this;
}

}  // namespace experimental

}  // namespace jet

#endif  // INCLUDE_JET_DETAIL_CUDA_ARRAY1_INL_H_

#endif  // JET_USE_CUDA
