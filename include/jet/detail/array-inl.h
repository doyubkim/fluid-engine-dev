// Copyright (c) 2018 Doyub Kim
//
// I am making my contributions/submissions to this project solely in my
// personal capacity and am not conveying any rights to any intellectual
// property of any third parties.

#ifndef INCLUDE_JET_DETAIL_ARRAY_INL_H_
#define INCLUDE_JET_DETAIL_ARRAY_INL_H_

#include <jet/array.h>
#include <jet/array_view.h>
#include <jet/iteration_utils.h>

namespace jet {

template <typename T, size_t N, typename Device>
class Array;

namespace internal {

template <typename T, size_t N, size_t I>
struct GetSizeAndInitVal {
    template <typename... Args>
    static void call(Vector<size_t, N>& size, T& value, size_t n,
                     Args... args) {
        size[N - I - 1] = n;
        GetSizeAndInitVal<T, N, I - 1>::call(size, value, args...);
    }
};

template <typename T, size_t N>
struct GetSizeAndInitVal<T, N, 0> {
    static void call(Vector<size_t, N>& size, T& value, size_t n) {
        call(size, value, n, T{});
    }

    static void call(Vector<size_t, N>& size, T& value, size_t n,
                     const T& initVal) {
        size[N - 1] = n;
        value = initVal;
    }
};

template <typename T, size_t N, size_t I>
struct GetSizeFromInitList {
    static size_t call(Vector<size_t, N>& size,
                       NestedInitializerListsT<T, I> lst) {
        size[I - 1] = lst.size();
        size_t i = 0;
        for (auto subLst : lst) {
            if (i == 0) {
                GetSizeFromInitList<T, N, I - 1>::call(size, subLst);
            } else {
                Vector<size_t, N> tempSizeN;
                size_t otherSize =
                    GetSizeFromInitList<T, N, I - 1>::call(tempSizeN, subLst);
                (void)otherSize;
                JET_ASSERT(otherSize == tempSizeN[I - 2]);
            }
            ++i;
        }
        return size[I - 1];
    }
};

template <typename T, size_t N>
struct GetSizeFromInitList<T, N, 1> {
    static size_t call(Vector<size_t, N>& size,
                       NestedInitializerListsT<T, 1> lst) {
        size[0] = lst.size();
        return size[0];
    }
};

template <typename T, size_t N, typename Device, size_t I>
struct SetArrayFromInitList {
    static void call(Array<T, N, Device>& arr,
                     NestedInitializerListsT<T, I> lst) {
        size_t i = 0;
        for (auto subLst : lst) {
            JET_ASSERT(i < arr.size()[I - 1]);
            SetArrayFromInitList<T, N, Device, I - 1>::call(arr, subLst, i);
            ++i;
        }
    }

    template <typename... RemainingIndices>
    static void call(Array<T, N, Device>& arr,
                     NestedInitializerListsT<T, I> lst,
                     RemainingIndices... indices) {
        size_t i = 0;
        for (auto subLst : lst) {
            JET_ASSERT(i < arr.size()[I - 1]);
            SetArrayFromInitList<T, N, Device, I - 1>::call(arr, subLst, i,
                                                            indices...);
            ++i;
        }
    }
};

template <typename T, size_t N, typename Device>
struct SetArrayFromInitList<T, N, Device, 1> {
    static void call(Array<T, N, Device>& arr,
                     NestedInitializerListsT<T, 1> lst) {
        size_t i = 0;
        for (auto val : lst) {
            JET_ASSERT(i < arr.size()[0]);
            arr(i) = val;
            ++i;
        }
    }

    template <typename... RemainingIndices>
    static void call(Array<T, N, Device>& arr,
                     NestedInitializerListsT<T, 1> lst,
                     RemainingIndices... indices) {
        size_t i = 0;
        for (auto val : lst) {
            JET_ASSERT(i < arr.size()[0]);
            arr(i, indices...) = val;
            ++i;
        }
    }
};

}  // namespace internal

// MARK: ArrayBase
template <typename T, size_t N, typename Device, typename Derived>
size_t ArrayBase<T, N, Device, Derived>::index(size_t i) const {
    return i;
}

template <typename T, size_t N, typename Device, typename Derived>
template <typename... Args>
size_t ArrayBase<T, N, Device, Derived>::index(size_t i, Args... args) const {
    static_assert(sizeof...(args) == N - 1, "Invalid number of indices.");
    return i + _size[0] * _index(1, args...);
}

template <typename T, size_t N, typename Device, typename Derived>
template <size_t... I>
size_t ArrayBase<T, N, Device, Derived>::index(
    const Vector<size_t, N>& idx) const {
    return _index(idx, std::make_index_sequence<N>{});
}

template <typename T, size_t N, typename Device, typename Derived>
T* ArrayBase<T, N, Device, Derived>::data() {
    return _handle.data();
}

template <typename T, size_t N, typename Device, typename Derived>
const T* ArrayBase<T, N, Device, Derived>::data() const {
    return _handle.data();
}

template <typename T, size_t N, typename Device, typename Derived>
const Vector<size_t, N>& ArrayBase<T, N, Device, Derived>::size() const {
    return _size;
}

template <typename T, size_t N, typename Device, typename Derived>
template <size_t M>
std::enable_if_t<(M > 0), size_t> ArrayBase<T, N, Device, Derived>::width()
    const {
    return _size[0];
}

template <typename T, size_t N, typename Device, typename Derived>
template <size_t M>
std::enable_if_t<(M > 1), size_t> ArrayBase<T, N, Device, Derived>::height()
    const {
    return _size[1];
}

template <typename T, size_t N, typename Device, typename Derived>
template <size_t M>
std::enable_if_t<(M > 2), size_t> ArrayBase<T, N, Device, Derived>::depth()
    const {
    return _size[2];
}

template <typename T, size_t N, typename Device, typename Derived>
size_t ArrayBase<T, N, Device, Derived>::length() const {
    return product<size_t, N>(_size, 1);
}

template <typename T, size_t N, typename Device, typename Derived>
typename ArrayBase<T, N, Device, Derived>::iterator
ArrayBase<T, N, Device, Derived>::begin() {
    return _handle.ptr;
}

template <typename T, size_t N, typename Device, typename Derived>
typename ArrayBase<T, N, Device, Derived>::const_iterator
ArrayBase<T, N, Device, Derived>::begin() const {
    return _handle.ptr;
}

template <typename T, size_t N, typename Device, typename Derived>
typename ArrayBase<T, N, Device, Derived>::iterator
ArrayBase<T, N, Device, Derived>::end() {
    return begin() + length();
}

template <typename T, size_t N, typename Device, typename Derived>
typename ArrayBase<T, N, Device, Derived>::const_iterator
ArrayBase<T, N, Device, Derived>::end() const {
    return begin() + length();
}

template <typename T, size_t N, typename Device, typename Derived>
typename ArrayBase<T, N, Device, Derived>::MemoryHandle
ArrayBase<T, N, Device, Derived>::handle() const {
    return _handle;
}

template <typename T, size_t N, typename Device, typename Derived>
typename ArrayBase<T, N, Device, Derived>::reference
ArrayBase<T, N, Device, Derived>::at(size_t i) {
    return _handle.ptr[i];
}

template <typename T, size_t N, typename Device, typename Derived>
typename ArrayBase<T, N, Device, Derived>::const_reference
ArrayBase<T, N, Device, Derived>::at(size_t i) const {
    return _handle.ptr[i];
}

template <typename T, size_t N, typename Device, typename Derived>
template <typename... Args>
typename ArrayBase<T, N, Device, Derived>::reference
ArrayBase<T, N, Device, Derived>::at(size_t i, Args... args) {
    return _handle.ptr[index(i, args...)];
}

template <typename T, size_t N, typename Device, typename Derived>
template <typename... Args>
typename ArrayBase<T, N, Device, Derived>::const_reference
ArrayBase<T, N, Device, Derived>::at(size_t i, Args... args) const {
    return _handle.ptr[index(i, args...)];
}

template <typename T, size_t N, typename Device, typename Derived>
typename ArrayBase<T, N, Device, Derived>::reference
ArrayBase<T, N, Device, Derived>::at(const Vector<size_t, N>& idx) {
    return _handle.ptr[index(idx)];
}

template <typename T, size_t N, typename Device, typename Derived>
typename ArrayBase<T, N, Device, Derived>::const_reference
ArrayBase<T, N, Device, Derived>::at(const Vector<size_t, N>& idx) const {
    return _handle.ptr[index(idx)];
}

template <typename T, size_t N, typename Device, typename Derived>
typename ArrayBase<T, N, Device, Derived>::reference
    ArrayBase<T, N, Device, Derived>::operator[](size_t i) {
    return at(i);
}

template <typename T, size_t N, typename Device, typename Derived>
typename ArrayBase<T, N, Device, Derived>::const_reference
    ArrayBase<T, N, Device, Derived>::operator[](size_t i) const {
    return at(i);
}

template <typename T, size_t N, typename Device, typename Derived>
template <typename... Args>
typename ArrayBase<T, N, Device, Derived>::reference
ArrayBase<T, N, Device, Derived>::operator()(size_t i, Args... args) {
    return at(i, args...);
}

template <typename T, size_t N, typename Device, typename Derived>
template <typename... Args>
typename ArrayBase<T, N, Device, Derived>::const_reference
ArrayBase<T, N, Device, Derived>::operator()(size_t i, Args... args) const {
    return at(i, args...);
}

template <typename T, size_t N, typename Device, typename Derived>
typename ArrayBase<T, N, Device, Derived>::reference
ArrayBase<T, N, Device, Derived>::operator()(const Vector<size_t, N>& idx) {
    return at(idx);
}

template <typename T, size_t N, typename Device, typename Derived>
typename ArrayBase<T, N, Device, Derived>::const_reference
ArrayBase<T, N, Device, Derived>::operator()(
    const Vector<size_t, N>& idx) const {
    return at(idx);
}

template <typename T, size_t N, typename Device, typename Derived>
ArrayBase<T, N, Device, Derived>::ArrayBase() : _size{} {}

template <typename T, size_t N, typename Device, typename Derived>
ArrayBase<T, N, Device, Derived>::ArrayBase(const ArrayBase& other) {
    setHandleAndSize(other._handle, other._size);
}

template <typename T, size_t N, typename Device, typename Derived>
ArrayBase<T, N, Device, Derived>::ArrayBase(ArrayBase&& other) {
    *this = std::move(other);
}

template <typename T, size_t N, typename Device, typename Derived>
template <typename... Args>
void ArrayBase<T, N, Device, Derived>::setHandleAndSize(MemoryHandle handle,
                                                        size_t ni,
                                                        Args... args) {
    setHandleAndSize(handle, Vector<size_t, N>{ni, args...});
}

template <typename T, size_t N, typename Device, typename Derived>
void ArrayBase<T, N, Device, Derived>::setHandleAndSize(
    MemoryHandle handle, Vector<size_t, N> size) {
    _handle = handle;
    _size = size;
}

template <typename T, size_t N, typename Device, typename Derived>
void ArrayBase<T, N, Device, Derived>::swapHandleAndSize(ArrayBase& other) {
    std::swap(_handle, other._handle);
    std::swap(_size, other._size);
}

template <typename T, size_t N, typename Device, typename Derived>
void ArrayBase<T, N, Device, Derived>::clear() {
    setHandleAndSize(MemoryHandle(), Vector<size_t, N>{});
}

template <typename T, size_t N, typename Device, typename Derived>
ArrayBase<T, N, Device, Derived>& ArrayBase<T, N, Device, Derived>::operator=(
    const ArrayBase& other) {
    setHandleAndSize(other._handle, other._size);
    return *this;
}

template <typename T, size_t N, typename Device, typename Derived>
ArrayBase<T, N, Device, Derived>& ArrayBase<T, N, Device, Derived>::operator=(
    ArrayBase&& other) {
    setHandleAndSize(other._handle, other._size);
    other.setHandleAndSize(MemoryHandle(), Vector<size_t, N>{});
    return *this;
}

template <typename T, size_t N, typename Device, typename Derived>
template <typename... Args>
size_t ArrayBase<T, N, Device, Derived>::_index(size_t d, size_t i,
                                                Args... args) const {
    return i + _size[d] * _index(d + 1, args...);
}

template <typename T, size_t N, typename Device, typename Derived>
size_t ArrayBase<T, N, Device, Derived>::_index(size_t, size_t i) const {
    return i;
}

template <typename T, size_t N, typename Device, typename Derived>
template <size_t... I>
size_t ArrayBase<T, N, Device, Derived>::_index(
    const Vector<size_t, N>& idx, std::index_sequence<I...>) const {
    return index(idx[I]...);
}

// MARK: Array

// CTOR
template <typename T, size_t N, typename Device>
Array<T, N, Device>::Array() : Base() {}

template <typename T, size_t N, typename Device>
Array<T, N, Device>::Array(const Vector<size_t, N>& size_, const T& initVal)
    : Array() {
    _data.resize(product<size_t, N>(size_, 1), initVal);
    Base::setHandleAndSize(Device::handleFromContainer(_data), size_);
}

template <typename T, size_t N, typename Device>
template <typename... Args>
Array<T, N, Device>::Array(size_t nx, Args... args) {
    Vector<size_t, N> newSize;
    T initVal;
    internal::GetSizeAndInitVal<T, N, N - 1>::call(newSize, initVal, nx,
                                                   args...);

    Array newArray(newSize, initVal);
    *this = std::move(newArray);
}

template <typename T, size_t N, typename Device>
Array<T, N, Device>::Array(NestedInitializerListsT<T, N> lst) {
    Vector<size_t, N> newSize;
    internal::GetSizeFromInitList<T, N, N>::call(newSize, lst);

    Array<T, N, CpuDevice<T>> newCpuArray(newSize);
    internal::SetArrayFromInitList<T, N, CpuDevice<T>, N>::call(newCpuArray,
                                                                lst);
    copyFrom(newCpuArray);
}

template <typename T, size_t N, typename Device>
template <size_t M>
Array<T, N, Device>::Array(
    const std::enable_if_t<(M == 1), std::vector<T>>& vec) {
    Array newArray(vec.size());
    Device::copy(vec, newArray);
    *this = std::move(newArray);
}

template <typename T, size_t N, typename Device>
template <typename OtherDevice, typename OtherDerived>
Array<T, N, Device>::Array(
    const ArrayBase<T, N, OtherDevice, OtherDerived>& other)
    : Array() {
    copyFrom(other);
}

template <typename T, size_t N, typename Device>
Array<T, N, Device>::Array(const Array& other) : Array() {
    copyFrom(other);
}

template <typename T, size_t N, typename Device>
Array<T, N, Device>::Array(Array&& other) : Array() {
    *this = std::move(other);
}

template <typename T, size_t N, typename Device>
template <typename OtherDevice, typename OtherDerived>
void Array<T, N, Device>::copyFrom(
    const ArrayBase<T, N, OtherDevice, OtherDerived>& other) {
    resize(other.size());
    Device::copy(other, *this);
}

template <typename T, size_t N, typename Device>
void Array<T, N, Device>::fill(const T& val) {
    Device::fill(*this, val);
}

template <typename T, size_t N, typename Device>
void Array<T, N, Device>::resize(Vector<size_t, N> newSize, const T& initVal) {
    Array newArray(newSize, initVal);
    Vector<size_t, N> minSize = min(_size, newArray._size);
    Device::copy(*this, minSize, newArray);
    *this = std::move(newArray);
}

template <typename T, size_t N, typename Device>
template <typename... Args>
void Array<T, N, Device>::resize(size_t nx, Args... args) {
    Vector<size_t, N> newSize;
    T initVal;
    internal::GetSizeAndInitVal<T, N, N - 1>::call(newSize, initVal, nx,
                                                   args...);

    resize(newSize, initVal);
}

template <typename T, size_t N, typename Device>
template <size_t M>
std::enable_if_t<(M == 1), void> Array<T, N, Device>::append(const T& val) {
    _data.push_back(val);
    Base::setHandleAndSize(_data.data(), _data.size());
}

template <typename T, size_t N, typename Device>
template <typename OtherDevice, typename OtherDerived, size_t M>
std::enable_if_t<(M == 1), void> Array<T, N, Device>::append(
    const ArrayBase<T, N, OtherDevice, OtherDerived>& extra) {
    size_t oldSize = length();
    resize(oldSize + extra.length());
    Device::copy(extra, 0, *this, oldSize);
}

template <typename T, size_t N, typename Device>
void Array<T, N, Device>::clear() {
    Base::clear();
    _data.clear();
}

template <typename T, size_t N, typename Device>
void Array<T, N, Device>::swap(Array& other) {
    Base::swapHandleAndSize(other);
    _data.swap(other._data);
}

template <typename T, size_t N, typename Device>
ArrayView<T, N, Device> Array<T, N, Device>::view() {
    return ArrayView<T, N, Device>(*this);
};

template <typename T, size_t N, typename Device>
ArrayView<const T, N, Device> Array<T, N, Device>::view() const {
    return ArrayView<const T, N, Device>(*this);
};

template <typename T, size_t N, typename Device>
Array<T, N, Device>& Array<T, N, Device>::operator=(const Array& other) {
    copyFrom(other);
    return *this;
}

template <typename T, size_t N, typename Device>
Array<T, N, Device>& Array<T, N, Device>::operator=(Array&& other) {
    swap(other);
    other.clear();
    return *this;
}

}  // namespace jet

#endif  // INCLUDE_JET_DETAIL_ARRAY_INL_H_
