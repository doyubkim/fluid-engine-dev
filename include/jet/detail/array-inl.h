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

template <typename T, size_t N>
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

template <typename T, size_t N, size_t I>
struct SetArrayFromInitList {
    static void call(Array<T, N>& arr, NestedInitializerListsT<T, I> lst) {
        size_t i = 0;
        for (auto subLst : lst) {
            JET_ASSERT(i < arr.size()[I - 1]);
            SetArrayFromInitList<T, N, I - 1>::call(arr, subLst, i);
            ++i;
        }
    }

    template <typename... RemainingIndices>
    static void call(Array<T, N>& arr, NestedInitializerListsT<T, I> lst,
                     RemainingIndices... indices) {
        size_t i = 0;
        for (auto subLst : lst) {
            JET_ASSERT(i < arr.size()[I - 1]);
            SetArrayFromInitList<T, N, I - 1>::call(arr, subLst, i, indices...);
            ++i;
        }
    }
};

template <typename T, size_t N>
struct SetArrayFromInitList<T, N, 1> {
    static void call(Array<T, N>& arr, NestedInitializerListsT<T, 1> lst) {
        size_t i = 0;
        for (auto val : lst) {
            JET_ASSERT(i < arr.size()[0]);
            arr(i) = val;
            ++i;
        }
    }

    template <typename... RemainingIndices>
    static void call(Array<T, N>& arr, NestedInitializerListsT<T, 1> lst,
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
template <typename T, size_t N, typename D>
size_t ArrayBase<T, N, D>::index(size_t i) const {
    return i;
}

template <typename T, size_t N, typename D>
template <typename... Args>
size_t ArrayBase<T, N, D>::index(size_t i, Args... args) const {
    static_assert(sizeof...(args) == N - 1, "Invalid number of indices.");
    return i + _size[0] * _index(1, args...);
}

template <typename T, size_t N, typename D>
template <size_t... I>
size_t ArrayBase<T, N, D>::index(const Vector<size_t, N>& idx) const {
    return _index(idx, std::make_index_sequence<N>{});
}

template <typename T, size_t N, typename D>
T* ArrayBase<T, N, D>::data() {
    return _ptr;
}

template <typename T, size_t N, typename D>
const T* ArrayBase<T, N, D>::data() const {
    return _ptr;
}

template <typename T, size_t N, typename D>
const Vector<size_t, N>& ArrayBase<T, N, D>::size() const {
    return _size;
}

template <typename T, size_t N, typename D>
template <size_t M>
std::enable_if_t<(M > 0), size_t> ArrayBase<T, N, D>::width() const {
    return _size.x;
}

template <typename T, size_t N, typename D>
template <size_t M>
std::enable_if_t<(M > 1), size_t> ArrayBase<T, N, D>::height() const {
    return _size.y;
}

template <typename T, size_t N, typename D>
template <size_t M>
std::enable_if_t<(M > 2), size_t> ArrayBase<T, N, D>::depth() const {
    return _size.z;
}

template <typename T, size_t N, typename D>
bool ArrayBase<T, N, D>::isEmpty() const {
    return length() == 0;
}

template <typename T, size_t N, typename D>
size_t ArrayBase<T, N, D>::length() const {
    return product<size_t, N>(_size, 1);
}

template <typename T, size_t N, typename D>
typename ArrayBase<T, N, D>::iterator ArrayBase<T, N, D>::begin() {
    return _ptr;
}

template <typename T, size_t N, typename D>
typename ArrayBase<T, N, D>::const_iterator ArrayBase<T, N, D>::begin() const {
    return _ptr;
}

template <typename T, size_t N, typename D>
typename ArrayBase<T, N, D>::iterator ArrayBase<T, N, D>::end() {
    return _ptr + length();
}

template <typename T, size_t N, typename D>
typename ArrayBase<T, N, D>::const_iterator ArrayBase<T, N, D>::end() const {
    return _ptr + length();
}

template <typename T, size_t N, typename D>
typename ArrayBase<T, N, D>::iterator ArrayBase<T, N, D>::rbegin() {
    return end() - 1;
}

template <typename T, size_t N, typename D>
typename ArrayBase<T, N, D>::const_iterator ArrayBase<T, N, D>::rbegin() const {
    return end() - 1;
}

template <typename T, size_t N, typename D>
typename ArrayBase<T, N, D>::iterator ArrayBase<T, N, D>::rend() {
    return begin() - 1;
}

template <typename T, size_t N, typename D>
typename ArrayBase<T, N, D>::const_iterator ArrayBase<T, N, D>::rend() const {
    return begin() - 1;
}

template <typename T, size_t N, typename D>
T& ArrayBase<T, N, D>::at(size_t i) {
    return _ptr[i];
}

template <typename T, size_t N, typename D>
const T& ArrayBase<T, N, D>::at(size_t i) const {
    return _ptr[i];
}

template <typename T, size_t N, typename D>
template <typename... Args>
T& ArrayBase<T, N, D>::at(size_t i, Args... args) {
    return data()[index(i, args...)];
}

template <typename T, size_t N, typename D>
template <typename... Args>
const T& ArrayBase<T, N, D>::at(size_t i, Args... args) const {
    return _ptr[index(i, args...)];
}

template <typename T, size_t N, typename D>
T& ArrayBase<T, N, D>::at(const Vector<size_t, N>& idx) {
    return data()[index(idx)];
}

template <typename T, size_t N, typename D>
const T& ArrayBase<T, N, D>::at(const Vector<size_t, N>& idx) const {
    return data()[index(idx)];
}

template <typename T, size_t N, typename D>
T& ArrayBase<T, N, D>::operator[](size_t i) {
    return at(i);
}

template <typename T, size_t N, typename D>
const T& ArrayBase<T, N, D>::operator[](size_t i) const {
    return at(i);
}

template <typename T, size_t N, typename D>
template <typename... Args>
T& ArrayBase<T, N, D>::operator()(size_t i, Args... args) {
    return at(i, args...);
}

template <typename T, size_t N, typename D>
template <typename... Args>
const T& ArrayBase<T, N, D>::operator()(size_t i, Args... args) const {
    return at(i, args...);
}

template <typename T, size_t N, typename D>
T& ArrayBase<T, N, D>::operator()(const Vector<size_t, N>& idx) {
    return at(idx);
}

template <typename T, size_t N, typename D>
const T& ArrayBase<T, N, D>::operator()(const Vector<size_t, N>& idx) const {
    return at(idx);
}

template <typename T, size_t N, typename D>
ArrayBase<T, N, D>::ArrayBase() : _size{} {}

template <typename T, size_t N, typename D>
ArrayBase<T, N, D>::ArrayBase(const ArrayBase& other) {
    setPtrAndSize(other._ptr, other._size);
}

template <typename T, size_t N, typename D>
ArrayBase<T, N, D>::ArrayBase(ArrayBase&& other) {
    *this = std::move(other);
}

template <typename T, size_t N, typename D>
template <typename... Args>
void ArrayBase<T, N, D>::setPtrAndSize(T* ptr, size_t ni, Args... args) {
    setPtrAndSize(ptr, Vector<size_t, N>{ni, args...});
}

template <typename T, size_t N, typename D>
void ArrayBase<T, N, D>::setPtrAndSize(T* data, Vector<size_t, N> size) {
    _ptr = data;
    _size = size;
}

template <typename T, size_t N, typename D>
void ArrayBase<T, N, D>::clearPtrAndSize() {
    setPtrAndSize(nullptr, Vector<size_t, N>{});
}

template <typename T, size_t N, typename D>
void ArrayBase<T, N, D>::swapPtrAndSize(ArrayBase& other) {
    std::swap(_ptr, other._ptr);
    std::swap(_size, other._size);
}

template <typename T, size_t N, typename D>
ArrayBase<T, N, D>& ArrayBase<T, N, D>::operator=(const ArrayBase& other) {
    setPtrAndSize(other.data(), other.size());
    return *this;
}

template <typename T, size_t N, typename D>
ArrayBase<T, N, D>& ArrayBase<T, N, D>::operator=(ArrayBase&& other) {
    setPtrAndSize(other.data(), other.size());
    other.setPtrAndSize(nullptr, Vector<size_t, N>{});
    return *this;
}

template <typename T, size_t N, typename D>
template <typename... Args>
size_t ArrayBase<T, N, D>::_index(size_t d, size_t i, Args... args) const {
    return i + _size[d] * _index(d + 1, args...);
}

template <typename T, size_t N, typename D>
size_t ArrayBase<T, N, D>::_index(size_t, size_t i) const {
    return i;
}

template <typename T, size_t N, typename D>
template <size_t... I>
size_t ArrayBase<T, N, D>::_index(const Vector<size_t, N>& idx,
                                  std::index_sequence<I...>) const {
    return index(idx[I]...);
}

// MARK: Array

// CTOR
template <typename T, size_t N>
Array<T, N>::Array() : Base() {}

template <typename T, size_t N>
Array<T, N>::Array(const Vector<size_t, N>& size_, const T& initVal) : Array() {
    _data.resize(product<size_t, N>(size_, 1), initVal);
    Base::setPtrAndSize(_data.data(), size_);
}

template <typename T, size_t N>
template <typename... Args>
Array<T, N>::Array(size_t nx, Args... args) {
    Vector<size_t, N> size_;
    T initVal;
    internal::GetSizeAndInitVal<T, N, N - 1>::call(size_, initVal, nx, args...);
    _data.resize(product<size_t, N>(size_, 1), initVal);
    Base::setPtrAndSize(_data.data(), size_);
}

template <typename T, size_t N>
Array<T, N>::Array(NestedInitializerListsT<T, N> lst) {
    Vector<size_t, N> newSize{};
    internal::GetSizeFromInitList<T, N, N>::call(newSize, lst);
    _data.resize(product<size_t, N>(newSize, 1));
    Base::setPtrAndSize(_data.data(), newSize);
    internal::SetArrayFromInitList<T, N, N>::call(*this, lst);
}

template <typename T, size_t N>
template <typename OtherDerived>
Array<T, N>::Array(const ArrayBase<T, N, OtherDerived>& other) : Array() {
    copyFrom(other);
}

template <typename T, size_t N>
template <typename OtherDerived>
Array<T, N>::Array(const ArrayBase<const T, N, OtherDerived>& other) : Array() {
    copyFrom(other);
}

template <typename T, size_t N>
Array<T, N>::Array(const Array& other) : Array() {
    copyFrom(other);
}

template <typename T, size_t N>
Array<T, N>::Array(Array&& other) : Array() {
    *this = std::move(other);
}

template <typename T, size_t N>
template <typename D>
void Array<T, N>::copyFrom(const ArrayBase<T, N, D>& other) {
    resize(other.size());
    forEachIndex(Vector<size_t, N>{}, other.size(),
                 [&](auto... idx) { this->at(idx...) = other(idx...); });
}

template <typename T, size_t N>
template <typename D>
void Array<T, N>::copyFrom(const ArrayBase<const T, N, D>& other) {
    resize(other.size());
    forEachIndex(Vector<size_t, N>{}, other.size(),
                 [&](auto... idx) { this->at(idx...) = other(idx...); });
}

template <typename T, size_t N>
void Array<T, N>::fill(const T& val) {
    std::fill(_data.begin(), _data.end(), val);
}

template <typename T, size_t N>
void Array<T, N>::resize(Vector<size_t, N> size_, const T& initVal) {
    Array newArray(size_, initVal);
    Vector<size_t, N> minSize = min(_size, newArray._size);
    forEachIndex(minSize,
                 [&](auto... idx) { newArray(idx...) = (*this)(idx...); });
    *this = std::move(newArray);
}

template <typename T, size_t N>
template <typename... Args>
void Array<T, N>::resize(size_t nx, Args... args) {
    Vector<size_t, N> size_;
    T initVal;
    internal::GetSizeAndInitVal<T, N, N - 1>::call(size_, initVal, nx, args...);

    resize(size_, initVal);
}

template <typename T, size_t N>
template <size_t M>
std::enable_if_t<(M == 1), void> Array<T, N>::append(const T& val) {
    _data.push_back(val);
    Base::setPtrAndSize(_data.data(), _data.size());
}

template <typename T, size_t N>
template <typename OtherDerived, size_t M>
std::enable_if_t<(M == 1), void> Array<T, N>::append(
    const ArrayBase<T, N, OtherDerived>& extra) {
    _data.insert(_data.end(), extra.begin(), extra.end());
    Base::setPtrAndSize(_data.data(), _data.size());
}

template <typename T, size_t N>
template <typename OtherDerived, size_t M>
std::enable_if_t<(M == 1), void> Array<T, N>::append(
    const ArrayBase<const T, N, OtherDerived>& extra) {
    _data.insert(_data.end(), extra.begin(), extra.end());
    Base::setPtrAndSize(_data.data(), _data.size());
}

template <typename T, size_t N>
void Array<T, N>::clear() {
    Base::clearPtrAndSize();
    _data.clear();
}

template <typename T, size_t N>
void Array<T, N>::swap(Array& other) {
    Base::swapPtrAndSize(other);
    std::swap(_data, other._data);
}

template <typename T, size_t N>
ArrayView<T, N> Array<T, N>::view() {
    return ArrayView<T, N>(*this);
};

template <typename T, size_t N>
ArrayView<const T, N> Array<T, N>::view() const {
    return ArrayView<const T, N>(*this);
};

template <typename T, size_t N>
template <typename OtherDerived>
Array<T, N>& Array<T, N>::operator=(
    const ArrayBase<T, N, OtherDerived>& other) {
    copyFrom(other);
    return *this;
}

template <typename T, size_t N>
template <typename OtherDerived>
Array<T, N>& Array<T, N>::operator=(
    const ArrayBase<const T, N, OtherDerived>& other) {
    copyFrom(other);
    return *this;
}

template <typename T, size_t N>
Array<T, N>& Array<T, N>::operator=(const Array& other) {
    copyFrom(other);
    return *this;
}

template <typename T, size_t N>
Array<T, N>& Array<T, N>::operator=(Array&& other) {
    _data = std::move(other._data);
    Base::setPtrAndSize(other.data(), other.size());
    other.setPtrAndSize(nullptr, Vector<size_t, N>{});

    return *this;
}

}  // namespace jet

#endif  // INCLUDE_JET_DETAIL_ARRAY_INL_H_
