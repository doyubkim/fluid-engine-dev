// Copyright (c) 2016 Doyub Kim

#include "vector2.h"
#include "vector3.h"

namespace jet
{

    template <typename T>
    inline Vector3<T> uniformSampleCone(T u1, T u2, const Vector3<T>& axis, T angle);

    template <typename T>
    inline Vector3<T> uniformSampleHemisphere(T u1, T u2, const Vector3<T>& normal);

    template <typename T>
    inline Vector3<T> cosineWeightedSampleHemisphere(T u1, T u2, const Vector3<T>& normal);

    template <typename T>
    inline Vector3<T> uniformSampleSphere(T u1, T u2);

    template <typename T>
    inline Vector2<T> uniformSampleDisk(T u1, T u2);

}

#include "detail/samplers-inl.h"
