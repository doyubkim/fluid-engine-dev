// Copyright (c) 2018 Doyub Kim
//
// I am making my contributions/submissions to this project solely in my
// personal capacity and am not conveying any rights to any intellectual
// property of any third parties.

#ifndef INCLUDE_JET_RAY3_H_
#define INCLUDE_JET_RAY3_H_

#include <jet/vector3.h>
#include <jet/ray.h>

namespace jet {

//!
//! \brief      Class for 2-D ray.
//!
//! \tparam     T     The value type.
//!
template <typename T>
class Ray<T, 3> final {
 public:
    static_assert(
        std::is_floating_point<T>::value,
        "Ray only can be instantiated with floating point types");

    //! The origin of the ray.
    Vector3<T> origin;

    //! The direction of the ray.
    Vector3<T> direction;

    //! Constructs an empty ray that points (1, 0, 0) from (0, 0, 0).
    Ray();

    //! Constructs a ray with given origin and riection.
    Ray(const Vector3<T>& newOrigin, const Vector3<T>& newDirection);

    //! Copy constructor.
    Ray(const Ray& other);

    //! Returns a point on the ray at distance \p t.
    Vector3<T> pointAt(T t) const;
};

//! Type alias for 3-D ray.
template <typename T> using Ray3 = Ray<T, 3>;

//! Float-type 3-D ray.
typedef Ray3<float> Ray3F;

//! Double-type 3-D ray.
typedef Ray3<double> Ray3D;

}  // namespace jet

#include "detail/ray3-inl.h"

#endif  // INCLUDE_JET_RAY3_H_
