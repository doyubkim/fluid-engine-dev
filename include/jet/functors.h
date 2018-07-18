// Copyright (c) 2018 Doyub Kim
//
// I am making my contributions/submissions to this project solely in my
// personal capacity and am not conveying any rights to any intellectual
// property of any third parties.

#ifndef INCLUDE_JET_FUNCTORS_H_
#define INCLUDE_JET_FUNCTORS_H_

#include <functional>
#include <limits>

namespace jet {

//! No-op operator.
template <typename T>
struct NoOp {
    constexpr T operator()(const T& a) const;
};

//! Type casting operator.
template <typename T, typename U>
struct TypeCast {
    constexpr U operator()(const T& a) const;
};

//! Performs std::ceil.
template <typename T>
struct Ceil {
    constexpr T operator()(const T& a) const;
};

//! Performs std::floor.
template <typename T>
struct Floor {
    constexpr T operator()(const T& a) const;
};

//! Square operator (a * a).
template <typename T>
struct Square {
    constexpr T operator()(const T& a) const;
};

//! Reverse minus operator.
template <typename T>
struct RMinus {
    constexpr T operator()(const T& a, const T& b) const;
};

//! Reverse divides operator.
template <typename T>
struct RDivides {
    constexpr T operator()(const T& a, const T& b) const;
};

//! Add-and-assign operator (+=).
template <typename T>
struct IAdd {
    void operator()(T& a, const T& b) const;
};

//! Subtract-and-assign operator (-=).
template <typename T>
struct ISub {
    void operator()(T& a, const T& b) const;
};

//! Multiply-and-assign operator (*=).
template <typename T>
struct IMul {
    void operator()(T& a, const T& b) const;
};

//! Divide-and-assign operator (/=).
template <typename T>
struct IDiv {
    void operator()(T& a, const T& b) const;
};

//! Takes minimum value.
template <typename T>
struct Min {
    constexpr T operator()(const T& a, const T& b) const;
};

//! Takes maximum value.
template <typename T>
struct Max {
    constexpr T operator()(const T& a, const T& b) const;
};

//! Takes absolute minimum value.
template <typename T>
struct AbsMin {
    constexpr T operator()(const T& a, const T& b) const;
};

//! Takes absolute maximum value.
template <typename T>
struct AbsMax {
    constexpr T operator()(const T& a, const T& b) const;
};

//! True if similar
template <typename T>
struct SimilarTo {
    double tol;
    constexpr SimilarTo(double tol_ = std::numeric_limits<double>::epsilon()) : tol(tol_) {}
    constexpr bool operator()(const T& a, const T& b) const;
};

//! Clamps the input value with low/high.
template <typename T>
struct Clamp {
    constexpr T operator()(const T& a, const T& low, const T& high) const;
};

}  // namespace jet

#include "detail/functors-inl.h"

#endif  // INCLUDE_JET_FUNCTORS_H_
