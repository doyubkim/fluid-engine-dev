// Copyright (c) 2018 Doyub Kim
//
// I am making my contributions/submissions to this project solely in my
// personal capacity and am not conveying any rights to any intellectual
// property of any third parties.

#ifndef INCLUDE_JET_DETAIL_CUDA_ARRAY_INL_H_
#define INCLUDE_JET_DETAIL_CUDA_ARRAY_INL_H_

#ifdef JET_USE_CUDA

#include <jet/_cuda_array.h>
#include <jet/_cuda_array_view.h>

namespace jet {

namespace internal {

template <typename T1, typename T2, size_t N, size_t I>
struct CudaBlockCopyHelper {
    template <typename... RemainingIndices>
    JET_CUDA_HOST_DEVICE static void call(const T1* src,
                                          const thrust::array<size_t, N>& size,
                                          T2* dst,
                                          RemainingIndices... indices) {
        for (size_t i = 0; i < size[I - 1]; ++i) {
            CudaBlockCopyHelper<T1, T2, N, I - 1>::call(src, size, dst, i,
                                                        indices...);
        }
    }
};

template <typename T1, typename T2, size_t N>
struct CudaBlockCopyHelper<T1, T2, N, 1> {
    template <typename... RemainingIndices>
    JET_CUDA_HOST_DEVICE static void call(const T1* src,
                                          const thrust::array<size_t, N>& size,
                                          T2* dst,
                                          RemainingIndices... indices) {
        for (size_t i = 0; i < size[0]; ++i) {
            dst[index(size, i, indices...)] = src[index(size, i, indices...)];
        }
    }

    template <typename... Args>
    JET_CUDA_HOST_DEVICE static size_t index(
        const thrust::array<size_t, N>& size, size_t i, Args... args) {
        return i + size[0] * _index(size, 1, args...);
    }

    template <typename... Args>
    JET_CUDA_HOST_DEVICE static size_t _index(
        const thrust::array<size_t, N>& size, size_t d, size_t i,
        Args... args) {
        return i + size[d] * _index(size, d + 1, args...);
    }

    JET_CUDA_HOST_DEVICE static size_t _index(
        const thrust::array<size_t, N>& size, size_t, size_t i) {
        return i;
    }
};

template <typename T1, typename T2, size_t N>
struct CudaBlockCopy {
    const T1* src;
    T2* dst;
    thrust::array<size_t, N> size;

    CudaBlockCopy(const T1* src_, thrust::array<size_t, N> size_, T2* dst_)
        : src(src_), dst(dst_) {}

    void operator()(size_t i) {
        CudaBlockCopyHelper<T1, T2, N, N - 1>::call(src, size, dst, i);
    }
};

template <typename T1, typename T2>
struct CudaBlockCopy<T1, T2, 1> {
    const T1* src;
    T2* dst;
    thrust::array<size_t, 1> size;

    CudaBlockCopy(const T1* src_, thrust::array<size_t, 1> size_, T2* dst_)
        : src(src_), dst(dst_) {}

    void operator()(size_t i) { dst[i] = src[i]; }
};

}  // namespace internal

////////////////////////////////////////////////////////////////////////////////
// MARK: CudaDevice

template <typename T>
template <typename T1, typename T2, size_t N, typename D1, typename D2>
void CudaDevice<T>::copy(const ArrayBase<T1, N, CudaDevice<T1>, D1>& src,
                         const thrust::array<size_t, N>& size,
                         ArrayBase<T2, N, CudaDevice<T2>, D2>& dst) {
    thrust::for_each(
        thrust::counting_iterator<size_t>(0),
        thrust::counting_iterator<size_t>(size[N - 1]),
        internal::CudaBlockCopy<T1, T2, N>(
            src.data(), thrust::array<size_t, N>(size), dst.data()));
}

////////////////////////////////////////////////////////////////////////////////
// MARK: ArrayBase

template <typename T, size_t N, typename Derived>
size_t ArrayBase<T, N, CudaDevice<T>, Derived>::index(size_t i) const {
    return i;
}

template <typename T, size_t N, typename Derived>
template <typename... Args>
size_t ArrayBase<T, N, CudaDevice<T>, Derived>::index(size_t i,
                                                      Args... args) const {
    static_assert(sizeof...(args) == N - 1, "Invalid number of indices.");
    return i + _size[0] * _index(1, args...);
}

template <typename T, size_t N, typename Derived>
template <size_t... I>
size_t ArrayBase<T, N, CudaDevice<T>, Derived>::index(
    const thrust::array<size_t, N>& idx) const {
    return _index(idx, std::make_index_sequence<N>{});
}

template <typename T, size_t N, typename Derived>
T* ArrayBase<T, N, CudaDevice<T>, Derived>::data() {
    return _handle.data();
}

template <typename T, size_t N, typename Derived>
const T* ArrayBase<T, N, CudaDevice<T>, Derived>::data() const {
    return _handle.data();
}

template <typename T, size_t N, typename Derived>
const thrust::array<size_t, N>& ArrayBase<T, N, CudaDevice<T>, Derived>::size()
    const {
    return _size;
}

template <typename T, size_t N, typename Derived>
template <size_t M>
std::enable_if_t<(M > 0), size_t>
ArrayBase<T, N, CudaDevice<T>, Derived>::width() const {
    return _size[0];
}

template <typename T, size_t N, typename Derived>
template <size_t M>
std::enable_if_t<(M > 1), size_t>
ArrayBase<T, N, CudaDevice<T>, Derived>::height() const {
    return _size[1];
}

template <typename T, size_t N, typename Derived>
template <size_t M>
std::enable_if_t<(M > 2), size_t>
ArrayBase<T, N, CudaDevice<T>, Derived>::depth() const {
    return _size[2];
}

template <typename T, size_t N, typename Derived>
size_t ArrayBase<T, N, CudaDevice<T>, Derived>::length() const {
    // TODO: Replace thrust::array with Vector
    // return product<size_t, N>(_size, 1);
    size_t l = _size[0];
    for (size_t i = 1; i < N; ++i) {
        l *= _size[i];
    }
    return l;
}

template <typename T, size_t N, typename Derived>
typename ArrayBase<T, N, CudaDevice<T>, Derived>::iterator
ArrayBase<T, N, CudaDevice<T>, Derived>::begin() {
    return _handle.ptr;
}

template <typename T, size_t N, typename Derived>
typename ArrayBase<T, N, CudaDevice<T>, Derived>::const_iterator
ArrayBase<T, N, CudaDevice<T>, Derived>::begin() const {
    return _handle.ptr;
}

template <typename T, size_t N, typename Derived>
typename ArrayBase<T, N, CudaDevice<T>, Derived>::iterator
ArrayBase<T, N, CudaDevice<T>, Derived>::end() {
    return begin() + length();
}

template <typename T, size_t N, typename Derived>
typename ArrayBase<T, N, CudaDevice<T>, Derived>::const_iterator
ArrayBase<T, N, CudaDevice<T>, Derived>::end() const {
    return begin() + length();
}

template <typename T, size_t N, typename Derived>
typename ArrayBase<T, N, CudaDevice<T>, Derived>::MemoryHandle
ArrayBase<T, N, CudaDevice<T>, Derived>::handle() const {
    return _handle;
}

template <typename T, size_t N, typename Derived>
typename ArrayBase<T, N, CudaDevice<T>, Derived>::reference
ArrayBase<T, N, CudaDevice<T>, Derived>::at(size_t i) {
    return _handle.ptr[i];
}

template <typename T, size_t N, typename Derived>
T ArrayBase<T, N, CudaDevice<T>, Derived>::at(size_t i) const {
    return _handle.ptr[i];
}

template <typename T, size_t N, typename Derived>
template <typename... Args>
typename ArrayBase<T, N, CudaDevice<T>, Derived>::reference
ArrayBase<T, N, CudaDevice<T>, Derived>::at(size_t i, Args... args) {
    return _handle.ptr[index(i, args...)];
}

template <typename T, size_t N, typename Derived>
template <typename... Args>
T ArrayBase<T, N, CudaDevice<T>, Derived>::at(size_t i, Args... args) const {
    return _handle.ptr[index(i, args...)];
}

template <typename T, size_t N, typename Derived>
typename ArrayBase<T, N, CudaDevice<T>, Derived>::reference
ArrayBase<T, N, CudaDevice<T>, Derived>::at(
    const thrust::array<size_t, N>& idx) {
    return _handle.ptr[index(idx)];
}

template <typename T, size_t N, typename Derived>
T ArrayBase<T, N, CudaDevice<T>, Derived>::at(
    const thrust::array<size_t, N>& idx) const {
    return _handle.ptr[index(idx)];
}

template <typename T, size_t N, typename Derived>
typename ArrayBase<T, N, CudaDevice<T>, Derived>::reference
    ArrayBase<T, N, CudaDevice<T>, Derived>::operator[](size_t i) {
    return at(i);
}

template <typename T, size_t N, typename Derived>
T ArrayBase<T, N, CudaDevice<T>, Derived>::operator[](size_t i) const {
    return at(i);
}

template <typename T, size_t N, typename Derived>
template <typename... Args>
typename ArrayBase<T, N, CudaDevice<T>, Derived>::reference
ArrayBase<T, N, CudaDevice<T>, Derived>::operator()(size_t i, Args... args) {
    return at(i, args...);
}

template <typename T, size_t N, typename Derived>
template <typename... Args>
T ArrayBase<T, N, CudaDevice<T>, Derived>::operator()(size_t i,
                                                      Args... args) const {
    return at(i, args...);
}

template <typename T, size_t N, typename Derived>
typename ArrayBase<T, N, CudaDevice<T>, Derived>::reference
ArrayBase<T, N, CudaDevice<T>, Derived>::operator()(
    const thrust::array<size_t, N>& idx) {
    return at(idx);
}

template <typename T, size_t N, typename Derived>
T ArrayBase<T, N, CudaDevice<T>, Derived>::operator()(
    const thrust::array<size_t, N>& idx) const {
    return at(idx);
}

template <typename T, size_t N, typename Derived>
ArrayBase<T, N, CudaDevice<T>, Derived>::ArrayBase() : _size{} {}

template <typename T, size_t N, typename Derived>
ArrayBase<T, N, CudaDevice<T>, Derived>::ArrayBase(const ArrayBase& other) {
    setHandleAndSize(other._handle, other._size);
}

template <typename T, size_t N, typename Derived>
ArrayBase<T, N, CudaDevice<T>, Derived>::ArrayBase(ArrayBase&& other) {
    *this = std::move(other);
}

template <typename T, size_t N, typename Derived>
template <typename... Args>
void ArrayBase<T, N, CudaDevice<T>, Derived>::setHandleAndSize(
    MemoryHandle handle, size_t ni, Args... args) {
    setHandleAndSize(handle, thrust::array<size_t, N>{ni, args...});
}

template <typename T, size_t N, typename Derived>
void ArrayBase<T, N, CudaDevice<T>, Derived>::setHandleAndSize(
    MemoryHandle handle, thrust::array<size_t, N> size) {
    _handle = handle;
    _size = size;
}

template <typename T, size_t N, typename Derived>
void ArrayBase<T, N, CudaDevice<T>, Derived>::swapHandleAndSize(
    ArrayBase& other) {
    thrust::swap(_handle, other._handle);
    thrust::swap(_size, other._size);
}

template <typename T, size_t N, typename Derived>
void ArrayBase<T, N, CudaDevice<T>, Derived>::clear() {
    setHandleAndSize(MemoryHandle(), thrust::array<size_t, N>{});
}

template <typename T, size_t N, typename Derived>
ArrayBase<T, N, CudaDevice<T>, Derived>&
ArrayBase<T, N, CudaDevice<T>, Derived>::operator=(const ArrayBase& other) {
    setHandleAndSize(other._handle, other._size);
    return *this;
}

template <typename T, size_t N, typename Derived>
ArrayBase<T, N, CudaDevice<T>, Derived>&
ArrayBase<T, N, CudaDevice<T>, Derived>::operator=(ArrayBase&& other) {
    setHandleAndSize(other._handle, other._size);
    other.setHandleAndSize(MemoryHandle(), thrust::array<size_t, N>{});
    return *this;
}

template <typename T, size_t N, typename Derived>
template <typename... Args>
size_t ArrayBase<T, N, CudaDevice<T>, Derived>::_index(size_t d, size_t i,
                                                       Args... args) const {
    return i + _size[d] * _index(d + 1, args...);
}

template <typename T, size_t N, typename Derived>
size_t ArrayBase<T, N, CudaDevice<T>, Derived>::_index(size_t, size_t i) const {
    return i;
}

template <typename T, size_t N, typename Derived>
template <size_t... I>
size_t ArrayBase<T, N, CudaDevice<T>, Derived>::_index(
    const thrust::array<size_t, N>& idx, std::index_sequence<I...>) const {
    return index(idx[I]...);
}

////////////////////////////////////////////////////////////////////////////////
// MARK: Array

template <typename T, size_t N>
Array<T, N, CudaDevice<T>>::Array() : Base() {}

template <typename T, size_t N>
Array<T, N, CudaDevice<T>>::Array(const thrust::array<size_t, N>& size_,
                                  const T& initVal)
    : Array() {
    // TODO: Replace thrust::array with Vector
    size_t l = size_[0];
    for (size_t i = 1; i < N; ++i) {
        l *= size_[i];
    }
    _data.resize(l, initVal);
    Base::setHandleAndSize(Device::handleFromContainer(_data), size_);
}

template <typename T, size_t N>
template <typename... Args>
Array<T, N, CudaDevice<T>>::Array(size_t nx, Args... args) {
    // TODO: Replace thrust::array with Vector
    Vector<size_t, N> newSizeV;
    T initVal;
    internal::GetSizeAndInitVal<T, N, N - 1>::call(newSizeV, initVal, nx,
                                                   args...);
    thrust::array<size_t, N> newSize(newSizeV);
    Array newArray(newSize, initVal);
    *this = std::move(newArray);
}

template <typename T, size_t N>
Array<T, N, CudaDevice<T>>::Array(NestedInitializerListsT<T, N> lst) {
    Vector<size_t, N> newSize;
    internal::GetSizeFromInitList<T, N, N>::call(newSize, lst);

    Array<T, N, CpuDevice<T>> newCpuArray(newSize);
    internal::SetArrayFromInitList<T, N, CpuDevice<T>, N>::call(newCpuArray,
                                                                lst);
    copyFrom(newCpuArray);
}

template <typename T, size_t N>
template <size_t M>
Array<T, N, CudaDevice<T>>::Array(
    const std::enable_if_t<(M == 1), std::vector<T>>& vec) {
    Array newArray(vec.size());
    Device::copy(vec, newArray);
    *this = std::move(newArray);
}

template <typename T, size_t N>
template <typename OtherDevice, typename OtherDerived>
Array<T, N, CudaDevice<T>>::Array(
    const ArrayBase<T, N, OtherDevice, OtherDerived>& other)
    : Array() {
    copyFrom(other);
}

template <typename T, size_t N>
Array<T, N, CudaDevice<T>>::Array(const Array& other) : Array() {
    copyFrom(other);
}

template <typename T, size_t N>
Array<T, N, CudaDevice<T>>::Array(Array&& other) : Array() {
    *this = std::move(other);
}

template <typename T, size_t N>
template <typename OtherDevice, typename OtherDerived>
void Array<T, N, CudaDevice<T>>::copyFrom(
    const ArrayBase<T, N, OtherDevice, OtherDerived>& other) {
    resize(other.size());
    Device::copy(other, *this);
}

template <typename T, size_t N>
void Array<T, N, CudaDevice<T>>::fill(const T& val) {
    Device::fill(*this, val);
}

template <typename T, size_t N>
void Array<T, N, CudaDevice<T>>::resize(thrust::array<size_t, N> newSize,
                                        const T& initVal) {
    Array newArray(newSize, initVal);
    // TODO: Replace with Vector
    thrust::array<size_t, N> minSize;
    for (size_t i = 0; i < N; ++i) {
        minSize[i] = std::min(_size[i], newArray._size[i]);
    }
    Device::copy(*this, minSize, newArray);
    *this = std::move(newArray);
}

template <typename T, size_t N>
template <typename... Args>
void Array<T, N, CudaDevice<T>>::resize(size_t nx, Args... args) {
    // TODO: Replace thrust::array with Vector
    Vector<size_t, N> newSizeV;
    T initVal;
    internal::GetSizeAndInitVal<T, N, N - 1>::call(newSizeV, initVal, nx,
                                                   args...);

    thrust::array<size_t, N> newSize(newSizeV);
    resize(newSize, initVal);
}

template <typename T, size_t N>
template <size_t M>
std::enable_if_t<(M == 1), void> Array<T, N, CudaDevice<T>>::append(
    const T& val) {
    _data.push_back(val);
    Base::setHandleAndSize(_data.data(), _data.size());
}

template <typename T, size_t N>
template <typename OtherDevice, typename OtherDerived, size_t M>
std::enable_if_t<(M == 1), void> Array<T, N, CudaDevice<T>>::append(
    const ArrayBase<T, N, OtherDevice, OtherDerived>& extra) {
    size_t oldSize = length();
    resize(oldSize + extra.length());
    Device::copy(extra, 0, *this, oldSize);
}

template <typename T, size_t N>
void Array<T, N, CudaDevice<T>>::clear() {
    Base::clear();
    _data.clear();
}

template <typename T, size_t N>
void Array<T, N, CudaDevice<T>>::swap(Array& other) {
    Base::swapHandleAndSize(other);
    _data.swap(other._data);
}

template <typename T, size_t N>
ArrayView<T, N, CudaDevice<T>> Array<T, N, CudaDevice<T>>::view() {
    return ArrayView<T, N, CudaDevice<T>>(*this);
};

template <typename T, size_t N>
ArrayView<const T, N, CudaDevice<T>> Array<T, N, CudaDevice<T>>::view() const {
    return ArrayView<const T, N, CudaDevice<T>>(*this);
};

template <typename T, size_t N>
Array<T, N, CudaDevice<T>>& Array<T, N, CudaDevice<T>>::operator=(
    const Array& other) {
    copyFrom(other);
    return *this;
}

template <typename T, size_t N>
Array<T, N, CudaDevice<T>>& Array<T, N, CudaDevice<T>>::operator=(
    Array&& other) {
    swap(other);
    other.clear();
    return *this;
}

}  // namespace jet

#endif  // JET_USE_CUDA

#endif  // INCLUDE_JET_DETAIL_CUDA_ARRAY_INL_H_
