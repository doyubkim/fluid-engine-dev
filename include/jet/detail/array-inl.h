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

template <typename T, size_t N, typename Handle>
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

template <typename T, size_t N, typename Handle, size_t I>
struct SetArrayFromInitList {
    static void call(Array<T, N, Handle>& arr,
                     NestedInitializerListsT<T, I> lst) {
        size_t i = 0;
        for (auto subLst : lst) {
            JET_ASSERT(i < arr.size()[I - 1]);
            SetArrayFromInitList<T, N, Handle, I - 1>::call(arr, subLst, i);
            ++i;
        }
    }

    template <typename... RemainingIndices>
    static void call(Array<T, N, Handle>& arr,
                     NestedInitializerListsT<T, I> lst,
                     RemainingIndices... indices) {
        size_t i = 0;
        for (auto subLst : lst) {
            JET_ASSERT(i < arr.size()[I - 1]);
            SetArrayFromInitList<T, N, Handle, I - 1>::call(arr, subLst, i,
                                                            indices...);
            ++i;
        }
    }
};

template <typename T, size_t N, typename Handle>
struct SetArrayFromInitList<T, N, Handle, 1> {
    static void call(Array<T, N, Handle>& arr,
                     NestedInitializerListsT<T, 1> lst) {
        size_t i = 0;
        for (auto val : lst) {
            JET_ASSERT(i < arr.size()[0]);
            arr(i) = val;
            ++i;
        }
    }

    template <typename... RemainingIndices>
    static void call(Array<T, N, Handle>& arr,
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
template <typename T, size_t N, typename Handle, typename Derived>
size_t ArrayBase<T, N, Handle, Derived>::index(size_t i) const {
    return i;
}

template <typename T, size_t N, typename Handle, typename Derived>
template <typename... Args>
size_t ArrayBase<T, N, Handle, Derived>::index(size_t i, Args... args) const {
    static_assert(sizeof...(args) == N - 1, "Invalid number of indices.");
    return i + _size[0] * _index(1, args...);
}

template <typename T, size_t N, typename Handle, typename Derived>
template <size_t... I>
size_t ArrayBase<T, N, Handle, Derived>::index(
    const Vector<size_t, N>& idx) const {
    return _index(idx, std::make_index_sequence<N>{});
}

template <typename T, size_t N, typename Handle, typename Derived>
T* ArrayBase<T, N, Handle, Derived>::data() {
    return _handle.data();
}

template <typename T, size_t N, typename Handle, typename Derived>
const T* ArrayBase<T, N, Handle, Derived>::data() const {
    return _handle.data();
}

template <typename T, size_t N, typename Handle, typename Derived>
const Vector<size_t, N>& ArrayBase<T, N, Handle, Derived>::size() const {
    return _size;
}

template <typename T, size_t N, typename Handle, typename Derived>
template <size_t M>
std::enable_if_t<(M > 0), size_t> ArrayBase<T, N, Handle, Derived>::width()
    const {
    return _size[0];
}

template <typename T, size_t N, typename Handle, typename Derived>
template <size_t M>
std::enable_if_t<(M > 1), size_t> ArrayBase<T, N, Handle, Derived>::height()
    const {
    return _size[1];
}

template <typename T, size_t N, typename Handle, typename Derived>
template <size_t M>
std::enable_if_t<(M > 2), size_t> ArrayBase<T, N, Handle, Derived>::depth()
    const {
    return _size[2];
}

template <typename T, size_t N, typename Handle, typename Derived>
size_t ArrayBase<T, N, Handle, Derived>::length() const {
    return product<size_t, N>(_size, 1);
}

template <typename T, size_t N, typename Handle, typename Derived>
typename ArrayBase<T, N, Handle, Derived>::iterator
ArrayBase<T, N, Handle, Derived>::begin() {
    return _handle.ptr;
}

template <typename T, size_t N, typename Handle, typename Derived>
typename ArrayBase<T, N, Handle, Derived>::const_iterator
ArrayBase<T, N, Handle, Derived>::begin() const {
    return _handle.ptr;
}

template <typename T, size_t N, typename Handle, typename Derived>
typename ArrayBase<T, N, Handle, Derived>::iterator
ArrayBase<T, N, Handle, Derived>::end() {
    return begin() + length();
}

template <typename T, size_t N, typename Handle, typename Derived>
typename ArrayBase<T, N, Handle, Derived>::const_iterator
ArrayBase<T, N, Handle, Derived>::end() const {
    return begin() + length();
}

template <typename T, size_t N, typename Handle, typename Derived>
Handle ArrayBase<T, N, Handle, Derived>::devicePtr() const {
    return _handle;
}

template <typename T, size_t N, typename Handle, typename Derived>
typename ArrayBase<T, N, Handle, Derived>::reference
ArrayBase<T, N, Handle, Derived>::at(size_t i) {
    return _handle.ptr[i];
}

template <typename T, size_t N, typename Handle, typename Derived>
typename ArrayBase<T, N, Handle, Derived>::const_reference
ArrayBase<T, N, Handle, Derived>::at(size_t i) const {
    return _handle.ptr[i];
}

template <typename T, size_t N, typename Handle, typename Derived>
template <typename... Args>
typename ArrayBase<T, N, Handle, Derived>::reference
ArrayBase<T, N, Handle, Derived>::at(size_t i, Args... args) {
    return _handle.ptr[index(i, args...)];
}

template <typename T, size_t N, typename Handle, typename Derived>
template <typename... Args>
typename ArrayBase<T, N, Handle, Derived>::const_reference
ArrayBase<T, N, Handle, Derived>::at(size_t i, Args... args) const {
    return _handle.ptr[index(i, args...)];
}

template <typename T, size_t N, typename Handle, typename Derived>
typename ArrayBase<T, N, Handle, Derived>::reference
ArrayBase<T, N, Handle, Derived>::at(const Vector<size_t, N>& idx) {
    return _handle.ptr[index(idx)];
}

template <typename T, size_t N, typename Handle, typename Derived>
typename ArrayBase<T, N, Handle, Derived>::const_reference
ArrayBase<T, N, Handle, Derived>::at(const Vector<size_t, N>& idx) const {
    return _handle.ptr[index(idx)];
}

template <typename T, size_t N, typename Handle, typename Derived>
typename ArrayBase<T, N, Handle, Derived>::reference
    ArrayBase<T, N, Handle, Derived>::operator[](size_t i) {
    return at(i);
}

template <typename T, size_t N, typename Handle, typename Derived>
typename ArrayBase<T, N, Handle, Derived>::const_reference
    ArrayBase<T, N, Handle, Derived>::operator[](size_t i) const {
    return at(i);
}

template <typename T, size_t N, typename Handle, typename Derived>
template <typename... Args>
typename ArrayBase<T, N, Handle, Derived>::reference
ArrayBase<T, N, Handle, Derived>::operator()(size_t i, Args... args) {
    return at(i, args...);
}

template <typename T, size_t N, typename Handle, typename Derived>
template <typename... Args>
typename ArrayBase<T, N, Handle, Derived>::const_reference
ArrayBase<T, N, Handle, Derived>::operator()(size_t i, Args... args) const {
    return at(i, args...);
}

template <typename T, size_t N, typename Handle, typename Derived>
typename ArrayBase<T, N, Handle, Derived>::reference
ArrayBase<T, N, Handle, Derived>::operator()(const Vector<size_t, N>& idx) {
    return at(idx);
}

template <typename T, size_t N, typename Handle, typename Derived>
typename ArrayBase<T, N, Handle, Derived>::const_reference
ArrayBase<T, N, Handle, Derived>::operator()(
    const Vector<size_t, N>& idx) const {
    return at(idx);
}

template <typename T, size_t N, typename Handle, typename Derived>
ArrayBase<T, N, Handle, Derived>::ArrayBase() : _size{} {}

template <typename T, size_t N, typename Handle, typename Derived>
ArrayBase<T, N, Handle, Derived>::ArrayBase(const ArrayBase& other) {
    setHandleAndSize(other._handle, other._size);
}

template <typename T, size_t N, typename Handle, typename Derived>
ArrayBase<T, N, Handle, Derived>::ArrayBase(ArrayBase&& other) {
    *this = std::move(other);
}

template <typename T, size_t N, typename Handle, typename Derived>
template <typename... Args>
void ArrayBase<T, N, Handle, Derived>::setHandleAndSize(Handle handle,
                                                        size_t ni,
                                                        Args... args) {
    setHandleAndSize(handle, Vector<size_t, N>{ni, args...});
}

template <typename T, size_t N, typename Handle, typename Derived>
void ArrayBase<T, N, Handle, Derived>::setHandleAndSize(
    Handle handle, Vector<size_t, N> size) {
    _handle = handle;
    _size = size;
}

template <typename T, size_t N, typename Handle, typename Derived>
void ArrayBase<T, N, Handle, Derived>::swapHandleAndSize(ArrayBase& other) {
    std::swap(_handle, other._handle);
    std::swap(_size, other._size);
}

template <typename T, size_t N, typename Handle, typename Derived>
void ArrayBase<T, N, Handle, Derived>::clear() {
    setHandleAndSize(Handle(), Vector<size_t, N>{});
}

template <typename T, size_t N, typename Handle, typename Derived>
ArrayBase<T, N, Handle, Derived>& ArrayBase<T, N, Handle, Derived>::operator=(
    const ArrayBase& other) {
    setHandleAndSize(other._handle, other._size);
    return *this;
}

template <typename T, size_t N, typename Handle, typename Derived>
ArrayBase<T, N, Handle, Derived>& ArrayBase<T, N, Handle, Derived>::operator=(
    ArrayBase&& other) {
    setHandleAndSize(other._handle, other._size);
    other.setHandleAndSize(Handle(), Vector<size_t, N>{});
    return *this;
}

template <typename T, size_t N, typename Handle, typename Derived>
template <typename... Args>
size_t ArrayBase<T, N, Handle, Derived>::_index(size_t d, size_t i,
                                                Args... args) const {
    return i + _size[d] * _index(d + 1, args...);
}

template <typename T, size_t N, typename Handle, typename Derived>
size_t ArrayBase<T, N, Handle, Derived>::_index(size_t, size_t i) const {
    return i;
}

template <typename T, size_t N, typename Handle, typename Derived>
template <size_t... I>
size_t ArrayBase<T, N, Handle, Derived>::_index(
    const Vector<size_t, N>& idx, std::index_sequence<I...>) const {
    return index(idx[I]...);
}

// MARK: Array

// CTOR
template <typename T, size_t N, typename Handle>
Array<T, N, Handle>::Array() : Base() {}

template <typename T, size_t N, typename Handle>
Array<T, N, Handle>::Array(const Vector<size_t, N>& size_, const T& initVal)
    : Array() {
    _data.resize(product<size_t, N>(size_, 1), initVal);
    Base::setHandleAndSize(Handle::handleFromContainer(_data), size_);
}

template <typename T, size_t N, typename Handle>
template <typename... Args>
Array<T, N, Handle>::Array(size_t nx, Args... args) {
    Vector<size_t, N> size_;
    T initVal;
    internal::GetSizeAndInitVal<T, N, N - 1>::call(size_, initVal, nx, args...);
    _data.resize(product<size_t, N>(size_, 1), initVal);
    Base::setHandleAndSize(Handle::handleFromContainer(_data), size_);
}

template <typename T, size_t N, typename Handle>
Array<T, N, Handle>::Array(NestedInitializerListsT<T, N> lst) {
    Vector<size_t, N> newSize{};
    internal::GetSizeFromInitList<T, N, N>::call(newSize, lst);
    _data.resize(product<size_t, N>(newSize, 1));
    Base::setHandleAndSize(Handle::handleFromContainer(_data), newSize);
    internal::SetArrayFromInitList<T, N, Handle, N>::call(*this, lst);
}

template <typename T, size_t N, typename Handle>
template <size_t M>
Array<T, N, Handle>::Array(
    const std::enable_if_t<(M == 1), std::vector<T>>& vec) {
    _data.resize(vec.size());
#ifdef JET_USE_CUDA
    thrust::copy(vec.begin(), vec.end(), begin());
#else
    std::copy(vec.begin(), vec.end(), begin());
#endif
    Base::setHandleAndSize(Handle::handleFromContainer(_data), {vec.size()});
}

#ifdef JET_USE_CUDA
template <typename T, size_t N, typename Handle>
template <size_t M>
Array<T, N, Handle>::Array(
    const std::enable_if_t<(M == 1), thrust::device_vector<T>>& vec) {
    _data.resize(vec.size());
    thrust::copy(vec.begin(), vec.end(), begin());
    Base::setHandleAndSize(Handle::handleFromContainer(_data), {vec.size()});
}

template <typename T, size_t N, typename Handle>
template <size_t M>
Array<T, N, Handle>::Array(
    const std::enable_if_t<(N == 1), thrust::host_vector<T>>& vec) {
    _data.resize(vec.size());
    thrust::copy(vec.begin(), vec.end(), begin());
    Base::setHandleAndSize(Handle::handleFromContainer(_data), {vec.size()});
}
#endif

template <typename T, size_t N, typename Handle>
template <typename OtherHandle, typename OtherDerived>
Array<T, N, Handle>::Array(
    const ArrayBase<T, N, OtherHandle, OtherDerived>& other)
    : Array() {
    copyFrom(other);
}

template <typename T, size_t N, typename Handle>
Array<T, N, Handle>::Array(const Array& other) : Array() {
    copyFrom(other);
}

template <typename T, size_t N, typename Handle>
Array<T, N, Handle>::Array(Array&& other) : Array() {
    *this = std::move(other);
}

template <typename T, size_t N, typename Handle>
template <typename OtherHandle, typename OtherDerived>
void Array<T, N, Handle>::copyFrom(
    const ArrayBase<T, N, OtherHandle, OtherDerived>& other) {
    resize(other.size());
#ifdef JET_USE_CUDA
    thrust::copy(other.begin(), other.end(), begin());
#else
    std::copy(other.begin(), other.end(), begin());
#endif
}

template <typename T, size_t N, typename Handle>
void Array<T, N, Handle>::fill(const T& val) {
#ifdef JET_USE_CUDA
    thrust::fill(_data.begin(), _data.end(), val);
#else
    std::fill(_data.begin(), _data.end(), val);
#endif
}

template <typename T, size_t N, typename Handle>
void Array<T, N, Handle>::resize(Vector<size_t, N> size_, const T& initVal) {
    Array newArray(size_, initVal);
    Vector<size_t, N> minSize = min(_size, newArray._size);
    // TODO: Better copy from CUDA
    forEachIndex(minSize,
                 [&](auto... idx) { newArray(idx...) = (*this)(idx...); });
    *this = std::move(newArray);
}

template <typename T, size_t N, typename Handle>
template <typename... Args>
void Array<T, N, Handle>::resize(size_t nx, Args... args) {
    Vector<size_t, N> size_;
    T initVal;
    internal::GetSizeAndInitVal<T, N, N - 1>::call(size_, initVal, nx, args...);

    resize(size_, initVal);
}

template <typename T, size_t N, typename Handle>
template <size_t M>
std::enable_if_t<(M == 1), void> Array<T, N, Handle>::append(const T& val) {
    _data.push_back(val);
    Base::setHandleAndSize(_data.data(), _data.size());
}

template <typename T, size_t N, typename Handle>
template <typename OtherHandle, typename OtherDerived, size_t M>
std::enable_if_t<(M == 1), void> Array<T, N, Handle>::append(
    const ArrayBase<T, N, OtherHandle, OtherDerived>& extra) {
    _data.insert(_data.end(), extra._data.begin(), extra._data.end());
    Base::setHandleAndSize(_data.data(), _data.size());
}

template <typename T, size_t N, typename Handle>
void Array<T, N, Handle>::clear() {
    Base::clear();
    _data.clear();
}

template <typename T, size_t N, typename Handle>
void Array<T, N, Handle>::swap(Array& other) {
    Base::swapHandleAndSize(other);
    std::swap(_data, other._data);
}

template <typename T, size_t N, typename Handle>
ArrayView<T, N, Handle> Array<T, N, Handle>::view() {
    return ArrayView<T, N, Handle>(*this);
};

template <typename T, size_t N, typename Handle>
ArrayView<const T, N, Handle> Array<T, N, Handle>::view() const {
    return ArrayView<const T, N, Handle>(*this);
};

template <typename T, size_t N, typename Handle>
Array<T, N, Handle>& Array<T, N, Handle>::operator=(const Array& other) {
    copyFrom(other);
    return *this;
}

template <typename T, size_t N, typename Handle>
Array<T, N, Handle>& Array<T, N, Handle>::operator=(Array&& other) {
    _data = std::move(other._data);
    Base::setHandleAndSize(other.devicePtr(), other.size());
    other.setHandleAndSize(Handle(), Vector<size_t, N>{});

    return *this;
}

}  // namespace jet

#endif  // INCLUDE_JET_DETAIL_ARRAY_INL_H_
