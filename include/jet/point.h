// Copyright (c) 2016 Doyub Kim

#ifndef INCLUDE_JET_POINT_H_
#define INCLUDE_JET_POINT_H_

#include <jet/macros.h>
#include <array>
#include <type_traits>

namespace jet {

//!
//! \brief Generic N-D point class.
//! \tparam T - Number type.
//! \tparam N - Dimension.
//!
template <typename T, size_t N>
class Point final {
 public:
    static_assert(
        N > 0, "Size of static-sized point should be greater than zero.");
    static_assert(
        std::is_arithmetic<T>::value,
        "Point only can be instantiated with arithmetic types");

    std::array<T, N> elements;

    Point();
    template <typename... Params>
    explicit Point(Params... params);
    template <typename U>
    explicit Point(const std::initializer_list<U>& lst);
    Point(const Point& other);

    const T& operator[](size_t i) const;
    T& operator[](size_t);

 private:
    template <typename... Params>
    void setAt(size_t i, T v, Params... params);

    void setAt(size_t i, T v);
};

}  // namespace jet

#include "detail/point-inl.h"

#endif  // INCLUDE_JET_POINT_H_

