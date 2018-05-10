// Copyright (c) 2018 Doyub Kim
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

template <typename T>
CudaArray<T, 1>::CudaArray() {}

template <typename T>
CudaArray<T, 1>::CudaArray(size_t size, const T& initVal) {
    resize(size, initVal);
}

template <typename T>
CudaArray<T, 1>::CudaArray(const ConstArrayView1<T>& view) {
    set(view);
}

template <typename T>
CudaArray<T, 1>::CudaArray(const ConstCudaArrayView<T, 1>& view) {
    set(view);
}

template <typename T>
CudaArray<T, 1>::CudaArray(const std::initializer_list<T>& lst) {
    set(lst);
}

template <typename T>
template <typename Alloc>
CudaArray<T, 1>::CudaArray(const std::vector<T, Alloc>& vec) {
    set(vec);
}

template <typename T>
template <typename Alloc>
CudaArray<T, 1>::CudaArray(const thrust::host_vector<T, Alloc>& vec) {
    set(vec);
}

template <typename T>
template <typename Alloc>
CudaArray<T, 1>::CudaArray(const thrust::device_vector<T, Alloc>& vec) {
    set(vec);
}

template <typename T>
CudaArray<T, 1>::CudaArray(const CudaArray& other) {
    set(other);
}

template <typename T>
CudaArray<T, 1>::CudaArray(CudaArray&& other) {
    (*this) = std::move(other);
}

template <typename T>
void CudaArray<T, 1>::set(const T& value) {
    thrust::fill(_data.begin(), _data.end(), value);
}

template <typename T>
void CudaArray<T, 1>::set(const ConstArrayView1<T>& view) {
    _data.resize(view.size());
    thrust::copy(view.begin(), view.end(), _data.begin());
}

template <typename T>
void CudaArray<T, 1>::set(const ConstCudaArrayView<T, 1>& view) {
    _data.resize(view.size());
    thrust::copy(view.begin(), view.end(), _data.begin());
}

template <typename T>
void CudaArray<T, 1>::set(const std::initializer_list<T>& lst) {
    Array1<T> temp(lst);
    set(temp.view());
}

template <typename T>
template <typename Alloc>
void CudaArray<T, 1>::set(const std::vector<T, Alloc>& vec) {
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
void CudaArray<T, 1>::set(const thrust::host_vector<T, Alloc>& vec) {
    _data = vec;
}

template <typename T>
template <typename Alloc>
void CudaArray<T, 1>::set(const thrust::device_vector<T, Alloc>& vec) {
    _data = vec;
}

template <typename T>
void CudaArray<T, 1>::set(const CudaArray& other) {
    _data = other._data;
}

template <typename T>
void CudaArray<T, 1>::clear() {
    _data.clear();
}

template <typename T>
void CudaArray<T, 1>::resize(size_t size, const T& initVal) {
    _data.resize(size, initVal);
}

template <typename T>
void CudaArray<T, 1>::swap(CudaArray& other) {
    _data.swap(other._data);
}

template <typename T>
size_t CudaArray<T, 1>::size() const {
    return _data.size();
}

template <typename T>
T* CudaArray<T, 1>::data() {
    return thrust::raw_pointer_cast(_data.data());
}

template <typename T>
const T* CudaArray<T, 1>::data() const {
    return thrust::raw_pointer_cast(_data.data());
}

template <typename T>
typename CudaArray<T, 1>::iterator CudaArray<T, 1>::begin() {
    return _data.data();
}

template <typename T>
typename CudaArray<T, 1>::const_iterator CudaArray<T, 1>::begin() const {
    return _data.data();
}

template <typename T>
typename CudaArray<T, 1>::iterator CudaArray<T, 1>::end() {
    return _data.data() + _data.size();
}

template <typename T>
typename CudaArray<T, 1>::const_iterator CudaArray<T, 1>::end() const {
    return _data.data() + _data.size();
}

template <typename T>
CudaArrayView<T, 1> CudaArray<T, 1>::view() {
    return CudaArrayView<T, 1>(*this);
}

template <typename T>
ConstCudaArrayView<T, 1> CudaArray<T, 1>::view() const {
    return CudaArrayView<T, 1>(*this);
}

template <typename T>
typename CudaArray<T, 1>::reference CudaArray<T, 1>::operator[](size_t i) {
    return _data[i];
}

template <typename T>
typename CudaArray<T, 1>::value_type CudaArray<T, 1>::operator[](
    size_t i) const {
    return _data[i];
}

template <typename T>
CudaArray<T, 1>& CudaArray<T, 1>::operator=(const T& value) {
    set(value);
    return *this;
}

template <typename T>
CudaArray<T, 1>& CudaArray<T, 1>::operator=(const ConstArrayView1<T>& view) {
    set(view);
    return *this;
}

template <typename T>
CudaArray<T, 1>& CudaArray<T, 1>::operator=(
    const ConstCudaArrayView<T, 1>& view) {
    set(view);
    return *this;
}

template <typename T>
template <typename Alloc>
CudaArray<T, 1>& CudaArray<T, 1>::operator=(const std::vector<T, Alloc>& vec) {
    set(vec);
    return *this;
}

template <typename T>
template <typename Alloc>
CudaArray<T, 1>& CudaArray<T, 1>::operator=(
    const thrust::host_vector<T, Alloc>& vec) {
    set(vec);
    return *this;
}

template <typename T>
template <typename Alloc>
CudaArray<T, 1>& CudaArray<T, 1>::operator=(
    const thrust::device_vector<T, Alloc>& vec) {
    set(vec);
    return *this;
}

template <typename T>
CudaArray<T, 1>& CudaArray<T, 1>::operator=(const CudaArray& other) {
    set(other);
    return *this;
}

template <typename T>
CudaArray<T, 1>& CudaArray<T, 1>::operator=(CudaArray&& other) {
    swap(other);
    other.clear();
    return *this;
}

}  // namespace jet

#endif  // INCLUDE_JET_DETAIL_CUDA_ARRAY1_INL_H_

#endif  // JET_USE_CUDA
