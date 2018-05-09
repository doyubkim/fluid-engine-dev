// Copyright (c) 2018 Doyub Kim
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

template <typename T>
CudaArrayView<T, 1>::CudaArrayView() {}

template <typename T>
CudaArrayView<T, 1>::CudaArrayView(T* data, size_t size) {
    set(data, size);
}

template <typename T>
CudaArrayView<T, 1>::CudaArrayView(const thrust::device_vector<T>& vec) {
    set(vec);
}

template <typename T>
CudaArrayView<T, 1>::CudaArrayView(const CudaArray<T, 1>& array) {
    set(array);
}

template <typename T>
CudaArrayView<T, 1>::CudaArrayView(const CudaArrayView<T, 1>& other) {
    set(other);
}

template <typename T>
CudaArrayView<T, 1>::CudaArrayView(CudaArrayView<T, 1>&& other) {
    *this = std::move(other);
}

template <typename T>
void CudaArrayView<T, 1>::set(pointer data, size_t size) {
    _data = thrust::device_pointer_cast<T>(data);
    _size = size;
}

template <typename T>
void CudaArrayView<T, 1>::set(const thrust::device_vector<T>& vec) {
    set(const_cast<T*>(thrust::raw_pointer_cast(vec.data())), vec.size());
}

template <typename T>
void CudaArrayView<T, 1>::set(const CudaArray<T, 1>& array) {
    set(const_cast<T*>(array.data()), array.size());
}

template <typename T>
void CudaArrayView<T, 1>::set(const CudaArrayView& other) {
    _data = other._data;
    _size = other._size;
}

template <typename T>
size_t CudaArrayView<T, 1>::size() const {
    return _size;
}

template <typename T>
typename CudaArrayView<T, 1>::pointer CudaArrayView<T, 1>::data() {
    return thrust::raw_pointer_cast(_data);
}

template <typename T>
typename CudaArrayView<T, 1>::const_pointer CudaArrayView<T, 1>::data() const {
    return thrust::raw_pointer_cast(_data);
}

template <typename T>
typename CudaArrayView<T, 1>::iterator CudaArrayView<T, 1>::begin() {
    return _data;
}

template <typename T>
typename CudaArrayView<T, 1>::const_iterator CudaArrayView<T, 1>::begin()
    const {
    return _data;
}

template <typename T>
typename CudaArrayView<T, 1>::iterator CudaArrayView<T, 1>::end() {
    return _data + _size;
}

template <typename T>
typename CudaArrayView<T, 1>::const_iterator CudaArrayView<T, 1>::end() const {
    return _data + _size;
}

template <typename T>
typename CudaArrayView<T, 1>::reference CudaArrayView<T, 1>::operator[](
    size_t i) {
    return _data[i];
}

template <typename T>
T CudaArrayView<T, 1>::operator[](size_t i) const {
    return _data[i];
}

template <typename T>
CudaArrayView<T, 1>& CudaArrayView<T, 1>::operator=(
    const thrust::device_vector<T>& vec) {
    set(vec);
    return *this;
}

template <typename T>
CudaArrayView<T, 1>& CudaArrayView<T, 1>::operator=(
    const CudaArray<T, 1>& array) {
    set(array);
    return *this;
}

template <typename T>
CudaArrayView<T, 1>& CudaArrayView<T, 1>::operator=(
    const CudaArrayView& other) {
    set(other);
    return *this;
}

template <typename T>
CudaArrayView<T, 1>& CudaArrayView<T, 1>::operator=(
    CudaArrayView<T, 1>&& view) {
    _data = view._data;
    _size = view._size;
    view._data = thrust::device_ptr<T>();
    view._size = 0;
    return *this;
}

//

template <typename T>
ConstCudaArrayView<T, 1>::ConstCudaArrayView() {}

template <typename T>
ConstCudaArrayView<T, 1>::ConstCudaArrayView(const_pointer data, size_t size) {
    set(data, size);
}

template <typename T>
ConstCudaArrayView<T, 1>::ConstCudaArrayView(
    const thrust::device_vector<T>& vec) {
    set(vec);
}

template <typename T>
ConstCudaArrayView<T, 1>::ConstCudaArrayView(const CudaArray<T, 1>& array) {
    set(array);
}

template <typename T>
ConstCudaArrayView<T, 1>::ConstCudaArrayView(const CudaArrayView<T, 1>& other) {
    set(other);
}

template <typename T>
ConstCudaArrayView<T, 1>::ConstCudaArrayView(const ConstCudaArrayView& other) {
    set(other);
}

template <typename T>
size_t ConstCudaArrayView<T, 1>::size() const {
    return _size;
}

template <typename T>
typename ConstCudaArrayView<T, 1>::const_pointer
ConstCudaArrayView<T, 1>::data() const {
    return thrust::raw_pointer_cast(_data);
}

template <typename T>
typename ConstCudaArrayView<T, 1>::const_iterator
ConstCudaArrayView<T, 1>::begin() const {
    return _data;
}

template <typename T>
typename ConstCudaArrayView<T, 1>::const_iterator
ConstCudaArrayView<T, 1>::end() const {
    return _data + _size;
}

template <typename T>
T ConstCudaArrayView<T, 1>::operator[](size_t i) const {
    return _data[i];
}

template <typename T>
ConstCudaArrayView<T, 1>& ConstCudaArrayView<T, 1>::operator=(
    const thrust::device_vector<T>& vec) {
    set(vec);
    return *this;
}

template <typename T>
ConstCudaArrayView<T, 1>& ConstCudaArrayView<T, 1>::operator=(
    const CudaArray<T, 1>& array) {
    set(array);
    return *this;
}

template <typename T>
ConstCudaArrayView<T, 1>& ConstCudaArrayView<T, 1>::operator=(
    const CudaArrayView<T, 1>& view) {
    set(view);
    return *this;
}

template <typename T>
ConstCudaArrayView<T, 1>& ConstCudaArrayView<T, 1>::operator=(
    const ConstCudaArrayView& other) {
    set(other);
    return *this;
}

template <typename T>
void ConstCudaArrayView<T, 1>::set(const_pointer data, size_t size) {
    _data = thrust::device_pointer_cast<T>(const_cast<T*>(data));
    _size = size;
}

template <typename T>
void ConstCudaArrayView<T, 1>::set(const thrust::device_vector<T>& vec) {
    set(const_cast<T*>(thrust::raw_pointer_cast(vec.data())), vec.size());
}

template <typename T>
void ConstCudaArrayView<T, 1>::set(const CudaArray<T, 1>& array) {
    set(array.data(), array.size());
}


template <typename T>
void ConstCudaArrayView<T, 1>::set(const CudaArrayView<T, 1>& view) {
    set(view.data(), view.size());
}

template <typename T>
void ConstCudaArrayView<T, 1>::set(const ConstCudaArrayView& other) {
    _data = other._data;
    _size = other._size;
}

}  // namespace jet

#endif  // INCLUDE_JET_DETAIL_CUDA_ARRAY_VIEW1_INL_H_

#endif  // JET_USE_CUDA
