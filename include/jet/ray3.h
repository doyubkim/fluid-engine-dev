// Copyright (c) 2016 Doyub Kim

#ifndef INCLUDE_JET_RAY3_H_
#define INCLUDE_JET_RAY3_H_

#include <jet/vector3.h>
#include <jet/ray.h>

namespace jet {

template <typename T>
class Ray<T, 3> final {
 public:
    static_assert(
        std::is_floating_point<T>::value,
        "Ray only can be instantiated with floating point types");

    Vector3<T> origin;
    Vector3<T> direction;

    Ray();

    Ray(const Vector3<T>& newOrigin, const Vector3<T>& newDirection);

    Ray(const Ray& other);

    Vector3<T> pointAt(T t) const;
};

template <typename T> using Ray3 = Ray<T, 3>;

typedef Ray3<float> Ray3F;
typedef Ray3<double> Ray3D;

}  // namespace jet

#include "detail/ray3-inl.h"

#endif  // INCLUDE_JET_RAY3_H_
