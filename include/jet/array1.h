// Copyright (c) 2017 Doyub Kim
//
// I am making my contributions/submissions to this project solely in my
// personal capacity and am not conveying any rights to any intellectual
// property of any third parties.

#ifndef INCLUDE_JET_ARRAY1_H_
#define INCLUDE_JET_ARRAY1_H_

#include <jet/array.h>
#include <jet/array_accessor1.h>

#include <fstream>
#include <functional>
#include <iostream>
#include <utility>  // just make cpplint happy..
#include <vector>

namespace jet {

//!
//! \brief 1-D array class.
//!
//! This class represents 1-D array data structure. This class is a simple
//! wrapper around std::vector with some additional features such as the array
//! accessor object and parallel for-loop.
//!
//! \tparam T - Type to store in the array.
//!
template <typename T>
class Array<T, 1> final {
 public:
    typedef std::vector<T> ContainerType;
    typedef typename ContainerType::iterator Iterator;
    typedef typename ContainerType::const_iterator ConstIterator;

    //! Constructs zero-sized 1-D array.
    Array();

    //! Constructs 1-D array with given \p size and fill it with \p initVal.
    //! \param size Initial size of the array.
    //! \param initVal Initial value of each array element.
    explicit Array(size_t size, const T& initVal = T());

    //!
    //! \brief Constructs 1-D array with given initializer list \p lst.
    //!
    //! This constructor will build 1-D array with given initializer list \p lst
    //! such as
    //!
    //! \code{.cpp}
    //! Array<int, 1> arr = {1, 2, 4, 9, 3};
    //! \endcode
    //!
    //! \param lst Initializer list that should be copy to the new array.
    //!
    Array(const std::initializer_list<T>& lst);

    //! Copy constructor.
    Array(const Array& other);

    //! Move constructor.
    Array(Array&& other);

    //! Sets entire array with given \p value.
    void set(const T& value);

    //! Copies given array \p other to this array.
    void set(const Array& other);

    //! Copies given initializer list \p lst to this array.
    void set(const std::initializer_list<T>& lst);

    //! Clears the array and resizes to zero.
    void clear();

    //! Resizes the array with \p size and fill the new element with \p initVal.
    void resize(size_t size, const T& initVal = T());

    //! Returns the reference to the i-th element.
    T& at(size_t i);

    //! Returns the const reference to the i-th element.
    const T& at(size_t i) const;

    //! Returns size of the array.
    size_t size() const;

    //! Returns the raw pointer to the array data.
    T* data();

    //! Returns the const raw pointer to the array data.
    const T* const data() const;

    //! Returns the begin iterator of the array.
    Iterator begin();

    //! Returns the begin const iterator of the array.
    ConstIterator begin() const;

    //! Returns the end iterator of the array.
    Iterator end();

    //! Returns the end const iterator of the array.
    ConstIterator end() const;

    //! Returns the array accessor.
    ArrayAccessor1<T> accessor();

    //! Returns the const array accessor.
    ConstArrayAccessor1<T> constAccessor() const;

    //! Swaps the content of the array with \p other array.
    void swap(Array& other);

    //! Appends single value \p newVal at the end of the array.
    void append(const T& newVal);

    //! Appends \p other array at the end of the array.
    void append(const Array& other);

    //!
    //! \brief Iterates the array and invoke given \p func for each element.
    //!
    //! This function iterates the array elements and invoke the callback
    //! function \p func. The callback function takes array's element as its
    //! input. The order of execution will be 0 to N-1 where N is the size of
    //! the array. Below is the sample usage:
    //!
    //! \code{.cpp}
    //! Array<int, 1> array(10, 4);
    //! array.forEach([](int elem) {
    //!     printf("%d\n", elem);
    //! });
    //! \endcode
    //!
    template <typename Callback>
    void forEach(Callback func) const;

    //!
    //! \brief Iterates the array and invoke given \p func for each index.
    //!
    //! This function iterates the array elements and invoke the callback
    //! function \p func. The callback function takes one parameter which is the
    //! index of the array. The order of execution will be 0 to N-1 where N is
    //! the size of the array. Below is the sample usage:
    //!
    //! \code{.cpp}
    //! Array<int, 1> array(10, 4);
    //! array.forEachIndex([&](size_t i) {
    //!     array[i] = 4.f * i + 1.5f;
    //! });
    //! \endcode
    //!
    template <typename Callback>
    void forEachIndex(Callback func) const;

    //!
    //! \brief Iterates the array and invoke given \p func for each element in
    //!     parallel using multi-threading.
    //!
    //! This function iterates the array elements and invoke the callback
    //! function \p func in parallel using multi-threading. The callback
    //! function takes array's element as its input. The order of execution will
    //! be non-deterministic since it runs in parallel.
    //! Below is the sample usage:
    //!
    //! \code{.cpp}
    //! Array<int, 1> array(1000, 4);
    //! array.parallelForEach([](int& elem) {
    //!     elem *= 2;
    //! });
    //! \endcode
    //!
    //! The parameter type of the callback function doesn't have to be T&, but
    //! const T& or T can be used as well.
    //!
    template <typename Callback>
    void parallelForEach(Callback func);

    //!
    //! \brief Iterates the array and invoke given \p func for each index in
    //!     parallel using multi-threading.
    //!
    //! This function iterates the array elements and invoke the callback
    //! function \p func in parallel using multi-threading. The callback
    //! function takes one parameter which is the index of the array. The order
    //! of execution will be non-deterministic since it runs in parallel.
    //! Below is the sample usage:
    //!
    //! \code{.cpp}
    //! Array<int, 1> array(1000, 4);
    //! array.parallelForEachIndex([](size_t i) {
    //!     array[i] *= 2;
    //! });
    //! \endcode
    //!
    template <typename Callback>
    void parallelForEachIndex(Callback func) const;

    //! Returns the reference to i-th element.
    T& operator[](size_t i);

    //! Returns the const reference to i-th element.
    const T& operator[](size_t i) const;

    //! Sets entire array with given \p value.
    Array& operator=(const T& other);

    //! Copies given array \p other to this array.
    Array& operator=(const Array& other);

    //! Move assignment.
    Array& operator=(Array&& other);

    //! Copies given initializer list \p lst to this array.
    Array& operator=(const std::initializer_list<T>& lst);

    //! Casts to array accessor.
    operator ArrayAccessor1<T>();

    //! Casts to const array accessor.
    operator ConstArrayAccessor1<T>() const;

 private:
    ContainerType _data;
};

//! Type alias for 1-D array.
template <typename T>
using Array1 = Array<T, 1>;

}  // namespace jet

#include "detail/array1-inl.h"

#endif  // INCLUDE_JET_ARRAY1_H_
