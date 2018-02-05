// Copyright (c) 2018 Doyub Kim
//
// I am making my contributions/submissions to this project solely in my
// personal capacity and am not conveying any rights to any intellectual
// property of any third parties.

#ifndef INCLUDE_JET_RAY2_H_
#define INCLUDE_JET_RAY2_H_

#include <jet/vector2.h>
#include <jet/ray.h>

namespace jet {

//!
//! \brief      Class for 2-D ray.
//!
//! \tparam     T     The value type.
//!
template <typename T>
class Ray<T, 2> final {
 public:
    static_assert(
        std::is_floating_point<T>::value,
        "Ray only can be instantiated with floating point types");

    //! The origin of the ray.
    Vector2<T> origin;

    //! The direction of the ray.
    Vector2<T> direction;

    //! Constructs an empty ray that points (1, 0) from (0, 0).
    Ray();

    //! Constructs a ray with given origin and riection.
    Ray(const Vector2<T>& newOrigin, const Vector2<T>& newDirection);

    //! Copy constructor.
    Ray(const Ray& other);

    //! Returns a point on the ray at distance \p t.
    Vector2<T> pointAt(T t) const;
};

//! Type alias for 2-D ray.
template <typename T> using Ray2 = Ray<T, 2>;

//! Float-type 2-D ray.
typedef Ray2<float> Ray2F;

//! Double-type 2-D ray.
typedef Ray2<double> Ray2D;

}  // namespace jet

#include "detail/ray2-inl.h"

#endif  // INCLUDE_JET_RAY2_H_
