// Copyright (c) 2018 Doyub Kim
//
// I am making my contributions/submissions to this project solely in my
// personal capacity and am not conveying any rights to any intellectual
// property of any third parties.

#ifndef INCLUDE_JET_ARRAY_VIEW2_H_
#define INCLUDE_JET_ARRAY_VIEW2_H_

#include <jet/array2.h>
#include <jet/array_view1.h>
#include <jet/size2.h>

namespace jet {

template <typename T>
class ArrayView<T, 2> final {
 public:
    typedef T value_type;
    typedef T& reference;
    typedef const T& const_reference;
    typedef T* pointer;
    typedef const T* const_pointer;
    typedef pointer iterator;
    typedef const_pointer const_iterator;

    ArrayView();

    ArrayView(pointer data, const Size2& size);

    ArrayView(const Array<T, 1>& array, const Size2& size);

    ArrayView(const Array<T, 2>& array);

    ArrayView(const std::vector<T>& vec, const Size2& size);

    ArrayView(const ArrayView<T, 1>& other, const Size2& size);

    ArrayView(const ArrayView& other);

    ArrayView(ArrayView&& other);

    void set(pointer data, const Size2& size);

    void set(const Array<T, 1>& array, const Size2& size);

    void set(const Array<T, 2>& array);

    void set(const std::vector<T>& vec, const Size2& size);

    void set(const ArrayView<T, 1>& other, const Size2& size);

    void set(const ArrayView& other);

    void swap(ArrayView& other);

    const Size2& size() const;

    size_t width() const;

    size_t height() const;

    pointer data();

    const_pointer data() const;

    pointer begin();

    const_pointer begin() const;

    pointer end();

    const_pointer end() const;

    //! Returns the reference to i-th element.
    reference operator[](size_t i);

    //! Returns the const reference to i-th element.
    const_reference operator[](size_t i) const;

    reference operator()(size_t i, size_t j);

    const_reference operator()(size_t i, size_t j) const;

    ArrayView& operator=(const Array<T, 2>& array);

    ArrayView& operator=(const ArrayView& other);

    ArrayView& operator=(ArrayView&& other);

 private:
    pointer _data = nullptr;
    Size2 _size{0, 0};
};

//! Type alias for 2-D array view.
template <typename T>
using ArrayView2 = ArrayView<T, 2>;

template <typename T>
class ConstArrayView<T, 2> final {
 public:
    typedef T value_type;
    typedef T& reference;
    typedef const T& const_reference;
    typedef T* pointer;
    typedef const T* const_pointer;
    typedef pointer iterator;
    typedef const_pointer const_iterator;

    ConstArrayView();

    ConstArrayView(const T* data, const Size2& size);

    ConstArrayView(const std::vector<T>& vec, const Size2& size);

    ConstArrayView(const Array<T, 1>& array, const Size2& size);

    ConstArrayView(const Array<T, 2>& array);

    ConstArrayView(const ArrayView<T, 1>& other, const Size2& size);

    ConstArrayView(const ArrayView<T, 2>& other);

    ConstArrayView(const ConstArrayView<T, 1>& other, const Size2& size);

    ConstArrayView(const ConstArrayView& other);

    ConstArrayView(ConstArrayView&& other);

    const Size2& size() const;

    size_t width() const;

    size_t height() const;

    const_pointer data() const;

    const_iterator begin() const;

    const_iterator end() const;

    //! Returns the const reference to i-th element.
    const_reference operator[](size_t i) const;

    const_reference operator()(size_t i, size_t j) const;

    ConstArrayView& operator=(const Array<T, 2>& array);

    ConstArrayView& operator=(const ArrayView<T, 2>& other);

    ConstArrayView& operator=(const ConstArrayView& other);

    ConstArrayView& operator=(ConstArrayView&& other);

 private:
    const_pointer _data = nullptr;
    Size2 _size{0, 0};

    void set(const T* data, const Size2& size);

    void set(const std::vector<T>& vec, const Size2& size);

    void set(const Array<T, 1>& array, const Size2& size);

    void set(const Array<T, 2>& array);

    void set(const ArrayView<T, 1>& other, const Size2& size);

    void set(const ArrayView<T, 2>& other);

    void set(const ConstArrayView<T, 1>& other, const Size2& size);

    void set(const ConstArrayView& other);
};

//! Type alias for const 2-D array view.
template <typename T>
using ConstArrayView2 = ConstArrayView<T, 2>;

}  // namespace jet

#include "detail/array_view2-inl.h"

#endif  // INCLUDE_JET_ARRAY_VIEW2_H_
