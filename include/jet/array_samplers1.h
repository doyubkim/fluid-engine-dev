// Copyright (c) 2016 Doyub Kim

#ifndef INCLUDE_JET_ARRAY_SAMPLERS1_H_
#define INCLUDE_JET_ARRAY_SAMPLERS1_H_

#include <jet/array_samplers.h>
#include <jet/array_accessor1.h>
#include <functional>

namespace jet {

template <typename T, typename R>
class NearestArraySampler<T, R, 1> final {
 public:
    static_assert(
        std::is_floating_point<R>::value,
        "Samplers only can be instantiated with floating point types");

    explicit NearestArraySampler(
        const ConstArrayAccessor1<T>& accessor,
        R gridSpacing,
        R gridOrigin);

    NearestArraySampler(const NearestArraySampler& other);

    T operator()(R pt) const;

    void getCoordinate(R x, size_t* i) const;

    std::function<T(R)> functor() const;

 private:
    R _gridSpacing;
    R _origin;
    ConstArrayAccessor1<T> _accessor;
};

template <typename T, typename R> using NearestArraySampler1
    = NearestArraySampler<T, R, 1>;


template <typename T, typename R>
class LinearArraySampler<T, R, 1> final {
 public:
    static_assert(
        std::is_floating_point<R>::value,
        "Samplers only can be instantiated with floating point types");

    explicit LinearArraySampler(
        const ConstArrayAccessor1<T>& accessor,
        R gridSpacing,
        R gridOrigin);

    LinearArraySampler(const LinearArraySampler& other);

    T operator()(R pt) const;

    void getCoordinatesAndWeights(
        R x, size_t* i0, size_t* i1, T* weight0, T* weight1) const;

    std::function<T(R)> functor() const;

 private:
    R _gridSpacing;
    R _origin;
    ConstArrayAccessor1<T> _accessor;
};

template <typename T, typename R> using LinearArraySampler1
    = LinearArraySampler<T, R, 1>;


template <typename T, typename R>
class CubicArraySampler<T, R, 1> final {
 public:
    static_assert(
        std::is_floating_point<R>::value,
        "Samplers only can be instantiated with floating point types");

    explicit CubicArraySampler(
        const ConstArrayAccessor1<T>& accessor,
        R gridSpacing,
        R gridOrigin);

    CubicArraySampler(const CubicArraySampler& other);

    T operator()(R pt) const;

    std::function<T(R)> functor() const;

 private:
    R _gridSpacing;
    R _origin;
    ConstArrayAccessor1<T> _accessor;
};

template <typename T, typename R> using CubicArraySampler1
    = CubicArraySampler<T, R, 1>;

}  // namespace jet

#include "detail/array_samplers1-inl.h"

#endif  // INCLUDE_JET_ARRAY_SAMPLERS1_H_
