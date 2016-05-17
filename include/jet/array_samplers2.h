// Copyright (c) 2016 Doyub Kim

#ifndef INCLUDE_JET_ARRAY_SAMPLERS2_H_
#define INCLUDE_JET_ARRAY_SAMPLERS2_H_

#include <jet/array_samplers.h>
#include <jet/array_accessor2.h>
#include <jet/vector2.h>
#include <functional>

namespace jet {

template <typename T, typename R>
class NearestArraySampler<T, R, 2> final {
 public:
    static_assert(
        std::is_floating_point<R>::value,
        "Samplers only can be instantiated with floating point types");

    explicit NearestArraySampler(
        const ConstArrayAccessor2<T>& accessor,
        const Vector2<R>& gridSpacing,
        const Vector2<R>& gridOrigin);

    NearestArraySampler(const NearestArraySampler& other);

    T operator()(const Vector2<R>& pt) const;

    void getCoordinate(const Vector2<R>& pt, Point2UI* index) const;

    std::function<T(const Vector2<R>&)> functor() const;

 private:
    Vector2<R> _gridSpacing;
    Vector2<R> _origin;
    ConstArrayAccessor2<T> _accessor;
};

template <typename T, typename R> using NearestArraySampler2
    = NearestArraySampler<T, R, 2>;


template <typename T, typename R>
class LinearArraySampler<T, R, 2> final {
 public:
    static_assert(
        std::is_floating_point<R>::value,
        "Samplers only can be instantiated with floating point types");

    explicit LinearArraySampler(
        const ConstArrayAccessor2<T>& accessor,
        const Vector2<R>& gridSpacing,
        const Vector2<R>& gridOrigin);

    LinearArraySampler(const LinearArraySampler& other);

    T operator()(const Vector2<R>& pt) const;

    void getCoordinatesAndWeights(
        const Vector2<R>& pt,
        std::array<Point2UI, 4>* indices,
        std::array<R, 4>* weights) const;

    std::function<T(const Vector2<R>&)> functor() const;

 private:
    Vector2<R> _gridSpacing;
    Vector2<R> _origin;
    ConstArrayAccessor2<T> _accessor;
};

template <typename T, typename R> using LinearArraySampler2
    = LinearArraySampler<T, R, 2>;


template <typename T, typename R>
class CubicArraySampler<T, R, 2> final {
 public:
    static_assert(
        std::is_floating_point<R>::value,
        "Samplers only can be instantiated with floating point types");

    explicit CubicArraySampler(
        const ConstArrayAccessor2<T>& accessor,
        const Vector2<R>& gridSpacing,
        const Vector2<R>& gridOrigin);

    CubicArraySampler(const CubicArraySampler& other);

    T operator()(const Vector2<R>& pt) const;

    std::function<T(const Vector2<R>&)> functor() const;

 private:
    Vector2<R> _gridSpacing;
    Vector2<R> _origin;
    ConstArrayAccessor2<T> _accessor;
};

template <typename T, typename R> using CubicArraySampler2
    = CubicArraySampler<T, R, 2>;

}  // namespace jet

#include "detail/array_samplers2-inl.h"

#endif  // INCLUDE_JET_ARRAY_SAMPLERS2_H_
