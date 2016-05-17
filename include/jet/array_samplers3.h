// Copyright (c) 2016 Doyub Kim

#ifndef INCLUDE_JET_ARRAY_SAMPLERS3_H_
#define INCLUDE_JET_ARRAY_SAMPLERS3_H_

#include <jet/array_samplers.h>
#include <jet/array_accessor3.h>
#include <jet/vector3.h>
#include <functional>

namespace jet {

template <typename T, typename R>
class NearestArraySampler<T, R, 3> final {
 public:
    static_assert(
        std::is_floating_point<R>::value,
        "Samplers only can be instantiated with floating point types");

    explicit NearestArraySampler(
        const ConstArrayAccessor3<T>& accessor,
        const Vector3<R>& gridSpacing,
        const Vector3<R>& gridOrigin);

    NearestArraySampler(const NearestArraySampler& other);

    T operator()(const Vector3<R>& pt) const;

    void getCoordinate(const Vector3<R>& pt, Point3UI* index) const;

    std::function<T(const Vector3<R>&)> functor() const;

 private:
    Vector3<R> _gridSpacing;
    Vector3<R> _origin;
    ConstArrayAccessor3<T> _accessor;
};

template <typename T, typename R> using NearestArraySampler3
    = NearestArraySampler<T, R, 3>;


template <typename T, typename R>
class LinearArraySampler<T, R, 3> final {
 public:
    static_assert(
        std::is_floating_point<R>::value,
        "Samplers only can be instantiated with floating point types");

    explicit LinearArraySampler(
        const ConstArrayAccessor3<T>& accessor,
        const Vector3<R>& gridSpacing,
        const Vector3<R>& gridOrigin);

    LinearArraySampler(const LinearArraySampler& other);

    T operator()(const Vector3<R>& pt) const;

    void getCoordinatesAndWeights(
        const Vector3<R>& pt,
        std::array<Point3UI, 8>* indices,
        std::array<R, 8>* weights) const;

    std::function<T(const Vector3<R>&)> functor() const;

 private:
    Vector3<R> _gridSpacing;
    Vector3<R> _origin;
    ConstArrayAccessor3<T> _accessor;
};

template <typename T, typename R> using LinearArraySampler3
    = LinearArraySampler<T, R, 3>;


template <typename T, typename R>
class CubicArraySampler<T, R, 3> final {
 public:
    static_assert(
        std::is_floating_point<R>::value,
        "Samplers only can be instantiated with floating point types");

    explicit CubicArraySampler(
        const ConstArrayAccessor3<T>& accessor,
        const Vector3<R>& gridSpacing,
        const Vector3<R>& gridOrigin);

    CubicArraySampler(const CubicArraySampler& other);

    T operator()(const Vector3<R>& pt) const;

    std::function<T(const Vector3<R>&)> functor() const;

 private:
    Vector3<R> _gridSpacing;
    Vector3<R> _origin;
    ConstArrayAccessor3<T> _accessor;
};

template <typename T, typename R> using CubicArraySampler3
    = CubicArraySampler<T, R, 3>;

}  // namespace jet

#include "detail/array_samplers3-inl.h"

#endif  // INCLUDE_JET_ARRAY_SAMPLERS3_H_
