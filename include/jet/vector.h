// Copyright (c) 2016 Doyub Kim

#ifndef INCLUDE_JET_VECTOR_H_
#define INCLUDE_JET_VECTOR_H_

#include <jet/constants.h>
#include <jet/type_helpers.h>
#include <array>
#include <type_traits>

namespace jet {

//!
//! \brief Generic N-D vector class.
//!
//! \tparam T - Number type.
//! \tparam N - Dimension.
//!

template <typename T, size_t N>
class Vector final {
 public:
    static_assert(
        N > 0,
        "Size of static-sized vector should be greater than zero.");
    static_assert(
        std::is_floating_point<T>::value,
        "Vector only can be instantiated with floating point types");

    //! Constructs a vector with zeros.
    Vector();

    //! Constructs vector instance with parameters.
    template <typename... Params>
    explicit Vector(Params... params);

    //! Constructs vector instance with initiazer list.
    template <typename U>
    Vector(const std::initializer_list<U>& lst);

    //! Copy constructor.
    Vector(const Vector& other);

    //! Set vector instance with initializer list.
    template <typename U>
    void set(const std::initializer_list<U>& lst);

    //! Set vector instance with other vector.
    void set(const Vector& other);

    //! Set vector instance with initializer list.
    template <typename U>
    Vector& operator=(const std::initializer_list<U>& lst);

    //! Set vector instance with other vector.
    Vector& operator=(const Vector& other);

    //! Returns the const reference to the \p i -th element.
    const T& operator[](size_t i) const;

    //! Returns the reference to the \p i -th element.
    T& operator[](size_t);

 private:
    std::array<T, N> _elements;

    template <typename... Params>
    void setAt(size_t i, T v, Params... params);
    void setAt(size_t i, T v);
};

//! Returns the type of the value.
template <typename T, size_t N>
struct ScalarType<Vector<T, N>> {
    typedef T value;
};

}  // namespace jet

#include "detail/vector-inl.h"

#endif  // INCLUDE_JET_VECTOR_H_

