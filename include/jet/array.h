// Copyright (c) 2018 Doyub Kim
//
// I am making my contributions/submissions to this project solely in my
// personal capacity and am not conveying any rights to any intellectual
// property of any third parties.

#ifndef INCLUDE_JET_ARRAY_H_
#define INCLUDE_JET_ARRAY_H_

#include <jet/matrix.h>
#include <jet/nested_initializer_list.h>

#include <algorithm>
#include <functional>
#include <vector>

namespace jet {

// MARK: ArrayBase

template <typename T, size_t N, typename DerivedArray>
class ArrayBase {
 public:
    using Derived = DerivedArray;
    using value_type = T;
    using reference = T&;
    using const_reference = const T&;
    using pointer = T*;
    using const_pointer = const T*;
    using iterator = T*;
    using const_iterator = const T*;

    virtual ~ArrayBase() = default;

    size_t index(size_t i) const;

    template <typename... Args>
    size_t index(size_t i, Args... args) const;

    template <size_t... I>
    size_t index(const Vector<size_t, N>& idx) const;

    pointer data();

    const_pointer data() const;

    const Vector<size_t, N>& size() const;

    template <size_t M = N>
    std::enable_if_t<(M > 0), size_t> width() const;

    template <size_t M = N>
    std::enable_if_t<(M > 1), size_t> height() const;

    template <size_t M = N>
    std::enable_if_t<(M > 2), size_t> depth() const;

    size_t length() const;

    iterator begin();

    const_iterator begin() const;

    iterator end();

    const_iterator end() const;

    reference at(size_t i);

    const_reference at(size_t i) const;

    template <typename... Args>
    reference at(size_t i, Args... args);

    template <typename... Args>
    const_reference at(size_t i, Args... args) const;

    reference at(const Vector<size_t, N>& idx);

    const_reference at(const Vector<size_t, N>& idx) const;

    reference operator[](size_t i);

    const_reference operator[](size_t i) const;

    template <typename... Args>
    reference operator()(size_t i, Args... args);

    template <typename... Args>
    const_reference operator()(size_t i, Args... args) const;

    reference operator()(const Vector<size_t, N>& idx);

    const_reference operator()(const Vector<size_t, N>& idx) const;

 protected:
    pointer _ptr = nullptr;
    Vector<size_t, N> _size;

    ArrayBase();

    ArrayBase(const ArrayBase& other);

    ArrayBase(ArrayBase&& other);

    template <typename... Args>
    void setPtrAndSize(pointer ptr, size_t ni, Args... args);

    void setPtrAndSize(pointer data, Vector<size_t, N> size);

    void swapPtrAndSize(ArrayBase& other);

    void clearPtrAndSize();

    ArrayBase& operator=(const ArrayBase& other);

    ArrayBase& operator=(ArrayBase&& other);

 private:
    template <typename... Args>
    size_t _index(size_t d, size_t i, Args... args) const;

    size_t _index(size_t, size_t i) const;

    template <size_t... I>
    size_t _index(const Vector<size_t, N>& idx,
                  std::index_sequence<I...>) const;
};

// MARK: Array

template <typename T, size_t N>
class ArrayView;

template <typename T, size_t N>
class Array final : public ArrayBase<T, N, Array<T, N>> {
    using Base = ArrayBase<T, N, Array<T, N>>;
    using Base::_size;
    using Base::setPtrAndSize;
    using Base::swapPtrAndSize;
    using Base::clearPtrAndSize;
    using Base::at;

 public:
    // CTOR
    Array();

    Array(const Vector<size_t, N>& size_, const T& initVal = T{});

    template <typename... Args>
    Array(size_t nx, Args... args);

    Array(NestedInitializerListsT<T, N> lst);

    Array(const Array& other);

    Array(Array&& other);

    template <typename D>
    void copyFrom(const ArrayBase<T, N, D>& other);

    void fill(const T& val);

    // resize
    void resize(Vector<size_t, N> size_, const T& initVal = T{});

    template <typename... Args>
    void resize(size_t nx, Args... args);

    template <size_t M = N>
    std::enable_if_t<(M == 1), void> append(const T& val);

    template <typename OtherDerived, size_t M = N>
    std::enable_if_t<(M == 1), void> append(
        const ArrayBase<T, N, OtherDerived>& extra);

    void clear();

    void swap(Array& other);

    // Views
    ArrayView<T, N> view();

    ArrayView<const T, N> view() const;

    // Assignment Operators
    template <typename OtherDerived>
    Array& operator=(const ArrayBase<T, N, OtherDerived>& other);

    Array& operator=(const Array& other);

    Array& operator=(Array&& other);

 private:
    std::vector<T> _data;
};

template <class T>
using Array1 = Array<T, 1>;

template <class T>
using Array2 = Array<T, 2>;

template <class T>
using Array3 = Array<T, 3>;

template <class T>
using Array4 = Array<T, 4>;

}  // namespace jet

#include <jet/detail/array-inl.h>

#endif  // INCLUDE_JET_ARRAY_H_
