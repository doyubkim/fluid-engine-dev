// Copyright (c) 2016 Doyub Kim

#ifndef INCLUDE_JET_SAMPLERS_H_
#define INCLUDE_JET_SAMPLERS_H_

#include <jet/vector2.h>
#include <jet/vector3.h>

namespace jet {

template <typename T>
inline Vector3<T> uniformSampleCone(
    T u1, T u2, const Vector3<T>& axis, T angle);

template <typename T>
inline Vector3<T> uniformSampleHemisphere(
    T u1, T u2, const Vector3<T>& normal);

template <typename T>
inline Vector3<T> cosineWeightedSampleHemisphere(
    T u1, T u2, const Vector3<T>& normal);

template <typename T>
inline Vector3<T> uniformSampleSphere(T u1, T u2);

template <typename T>
inline Vector2<T> uniformSampleDisk(T u1, T u2);

}  // namespace jet

#include "detail/samplers-inl.h"

#endif  // INCLUDE_JET_SAMPLERS_H_
