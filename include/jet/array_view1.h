// Copyright (c) 2017 Doyub Kim
//
// I am making my contributions/submissions to this project solely in my
// personal capacity and am not conveying any rights to any intellectual
// property of any third parties.

#ifndef INCLUDE_JET_ARRAY_VIEW1_H_
#define INCLUDE_JET_ARRAY_VIEW1_H_

#include <jet/array1.h>

namespace jet {

namespace experimental {

template <typename T>
class ArrayView1 final {
 public:
    ArrayView1();

    explicit ArrayView1(T* data, size_t size);

    ArrayView1(const Array1<T>& array);

    ArrayView1(const std::vector<T>& vec);

    ArrayView1(const ArrayView1& other);

    ArrayView1(ArrayView1&& other);

    void set(const T& value);

    void set(T* data, size_t size);

    void set(const Array1<T>& array);

    void set(const std::vector<T>& vec);

    void set(const ArrayView1& other);

    void swap(ArrayView1& other);

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

    ArrayView1& operator=(const T& value);

    ArrayView1& operator=(const std::vector<T>& vec);

    ArrayView1& operator=(const Array1<T>& array);

    ArrayView1& operator=(const ArrayView1& other);

    ArrayView1& operator=(ArrayView1&& other);

 private:
    T* _data = nullptr;
    size_t _size = 0;
};

}  // namespace experimental

}  // namespace jet

#include "detail/array_view1-inl.h"

#endif  // INCLUDE_JET_ARRAY_VIEW1_H_
