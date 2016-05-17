// Copyright (c) 2016 Doyub Kim

#ifndef INCLUDE_JET_RAY2_H_
#define INCLUDE_JET_RAY2_H_

#include <jet/vector2.h>
#include <jet/ray.h>

namespace jet {

template <typename T>
class Ray<T, 2> final {
 public:
    static_assert(
        std::is_floating_point<T>::value,
        "Ray only can be instantiated with floating point types");

    Vector2<T> origin;
    Vector2<T> direction;

    Ray();

    Ray(const Vector2<T>& newOrigin, const Vector2<T>& newDirection);

    Ray(const Ray& other);

    Vector2<T> pointAt(T t) const;
};

template <typename T> using Ray2 = Ray<T, 2>;

typedef Ray2<float> Ray2F;
typedef Ray2<double> Ray2D;

}  // namespace jet

#include "detail/ray2-inl.h"

#endif  // INCLUDE_JET_RAY2_H_
