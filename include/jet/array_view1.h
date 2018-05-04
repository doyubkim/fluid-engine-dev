// Copyright (c) 2018 Doyub Kim
//
// I am making my contributions/submissions to this project solely in my
// personal capacity and am not conveying any rights to any intellectual
// property of any third parties.

#ifndef INCLUDE_JET_ARRAY_VIEW1_H_
#define INCLUDE_JET_ARRAY_VIEW1_H_

#include <jet/array1.h>
#include <jet/array_view.h>

namespace jet {

template <typename T>
class ArrayView<T, 1> final {
 public:
    ArrayView();

    explicit ArrayView(T* data, size_t size);

    ArrayView(const std::vector<T>& vec);

    ArrayView(const Array1<T>& array);

    ArrayView(const ArrayView& other);

    ArrayView(ArrayView&& other);

    void set(T* data, size_t size);

    void set(const Array1<T>& array);

    void set(const std::vector<T>& vec);

    void set(const ArrayView& other);

    void swap(ArrayView& other);

    size_t size() const;

    T* data();

    const T* data() const;

    T* begin();

    const T* begin() const;

    T* end();

    const T* end() const;

    //! Returns the reference to i-th element.
    T& operator[](size_t i);

    //! Returns the const reference to i-th element.
    const T& operator[](size_t i) const;

    ArrayView& operator=(const std::vector<T>& vec);

    ArrayView& operator=(const Array1<T>& array);

    ArrayView& operator=(const ArrayView& other);

    ArrayView& operator=(ArrayView&& other);

 private:
    T* _data = nullptr;
    size_t _size = 0;
};

//! Type alias for 1-D array view.
template <typename T>
using ArrayView1 = ArrayView<T, 1>;

template <typename T>
class ConstArrayView<T, 1> final {
 public:
    ConstArrayView();

    explicit ConstArrayView(const T* data, size_t size);

    ConstArrayView(const std::vector<T>& vec);

    ConstArrayView(const Array1<T>& array);

    ConstArrayView(const ArrayView<T, 1>& other);

    ConstArrayView(const ConstArrayView& other);

    ConstArrayView(ConstArrayView&& other);

    void swap(ConstArrayView& other);

    size_t size() const;

    const T* data() const;

    const T* begin() const;

    const T* end() const;

    //! Returns the const reference to i-th element.
    const T& operator[](size_t i) const;

    ConstArrayView& operator=(const std::vector<T>& vec);

    ConstArrayView& operator=(const Array1<T>& array);

    ConstArrayView& operator=(const ArrayView<T, 1>& other);

    ConstArrayView& operator=(const ConstArrayView& other);

    ConstArrayView& operator=(ConstArrayView&& other);

 private:
    const T* _data = nullptr;
    size_t _size = 0;

    void set(const T* data, size_t size);

    void set(const std::vector<T>& vec);

    void set(const Array1<T>& array);

    void set(const ArrayView<T, 1>& other);

    void set(const ConstArrayView& other);
};

//! Type alias for const 1-D array view.
template <typename T>
using ConstArrayView1 = ConstArrayView<T, 1>;

}  // namespace jet

#include "detail/array_view1-inl.h"

#endif  // INCLUDE_JET_ARRAY_VIEW1_H_
