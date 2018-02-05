// Copyright (c) 2018 Doyub Kim
//
// I am making my contributions/submissions to this project solely in my
// personal capacity and am not conveying any rights to any intellectual
// property of any third parties.

#ifndef INCLUDE_JET_ARRAY_SAMPLERS3_H_
#define INCLUDE_JET_ARRAY_SAMPLERS3_H_

#include <jet/array_samplers.h>
#include <jet/array_accessor3.h>
#include <jet/vector3.h>
#include <functional>

namespace jet {

//!
//! \brief 3-D nearest array sampler class.
//!
//! This class provides nearest sampling interface for a given 3-D array.
//!
//! \tparam T - The value type to sample.
//! \tparam R - The real number type.
//!
template <typename T, typename R>
class NearestArraySampler<T, R, 3> final {
 public:
    static_assert(
        std::is_floating_point<R>::value,
        "Samplers only can be instantiated with floating point types");

    //!
    //! \brief      Constructs a sampler using array accessor, spacing between
    //!     the elements, and the position of the first array element.
    //!
    //! \param[in]  accessor    The array accessor.
    //! \param[in]  gridSpacing The grid spacing.
    //! \param[in]  gridOrigin  The grid origin.
    //!
    explicit NearestArraySampler(
        const ConstArrayAccessor3<T>& accessor,
        const Vector3<R>& gridSpacing,
        const Vector3<R>& gridOrigin);

    //! Copy constructor.
    NearestArraySampler(const NearestArraySampler& other);

    //! Returns sampled value at point \p pt.
    T operator()(const Vector3<R>& pt) const;

    //! Returns the nearest array index for point \p x.
    void getCoordinate(const Vector3<R>& pt, Point3UI* index) const;

    //! Returns a funtion object that wraps this instance.
    std::function<T(const Vector3<R>&)> functor() const;

 private:
    Vector3<R> _gridSpacing;
    Vector3<R> _origin;
    ConstArrayAccessor3<T> _accessor;
};

//! Type alias for 3-D nearest array sampler.
template <typename T, typename R> using NearestArraySampler3
    = NearestArraySampler<T, R, 3>;


//!
//! \brief 2-D linear array sampler class.
//!
//! This class provides linear sampling interface for a given 2-D array.
//!
//! \tparam T - The value type to sample.
//! \tparam R - The real number type.
//!
template <typename T, typename R>
class LinearArraySampler<T, R, 3> final {
 public:
    static_assert(
        std::is_floating_point<R>::value,
        "Samplers only can be instantiated with floating point types");

    //!
    //! \brief      Constructs a sampler using array accessor, spacing between
    //!     the elements, and the position of the first array element.
    //!
    //! \param[in]  accessor    The array accessor.
    //! \param[in]  gridSpacing The grid spacing.
    //! \param[in]  gridOrigin  The grid origin.
    //!
    explicit LinearArraySampler(
        const ConstArrayAccessor3<T>& accessor,
        const Vector3<R>& gridSpacing,
        const Vector3<R>& gridOrigin);

    //! Copy constructor.
    LinearArraySampler(const LinearArraySampler& other);

    //! Returns sampled value at point \p pt.
    T operator()(const Vector3<R>& pt) const;

    //! Returns the indices of points and their sampling weight for given point.
    void getCoordinatesAndWeights(
        const Vector3<R>& pt,
        std::array<Point3UI, 8>* indices,
        std::array<R, 8>* weights) const;

    //! Returns the indices of points and their gradient of sampling weight for
    //! given point.
    void getCoordinatesAndGradientWeights(
        const Vector3<R>& pt,
        std::array<Point3UI, 8>* indices,
        std::array<Vector3<R>, 8>* weights) const;

    //! Returns a funtion object that wraps this instance.
    std::function<T(const Vector3<R>&)> functor() const;

 private:
    Vector3<R> _gridSpacing;
    Vector3<R> _invGridSpacing;
    Vector3<R> _origin;
    ConstArrayAccessor3<T> _accessor;
};

//! Type alias for 3-D linear array sampler.
template <typename T, typename R> using LinearArraySampler3
    = LinearArraySampler<T, R, 3>;


//!
//! \brief 3-D cubic array sampler class.
//!
//! This class provides cubic sampling interface for a given 3-D array.
//!
//! \tparam T - The value type to sample.
//! \tparam R - The real number type.
//!
template <typename T, typename R>
class CubicArraySampler<T, R, 3> final {
 public:
    static_assert(
        std::is_floating_point<R>::value,
        "Samplers only can be instantiated with floating point types");

    //!
    //! \brief      Constructs a sampler using array accessor, spacing between
    //!     the elements, and the position of the first array element.
    //!
    //! \param[in]  accessor    The array accessor.
    //! \param[in]  gridSpacing The grid spacing.
    //! \param[in]  gridOrigin  The grid origin.
    //!
    explicit CubicArraySampler(
        const ConstArrayAccessor3<T>& accessor,
        const Vector3<R>& gridSpacing,
        const Vector3<R>& gridOrigin);

    //! Copy constructor.
    CubicArraySampler(const CubicArraySampler& other);

    //! Returns sampled value at point \p pt.
    T operator()(const Vector3<R>& pt) const;

    //! Returns a funtion object that wraps this instance.
    std::function<T(const Vector3<R>&)> functor() const;

 private:
    Vector3<R> _gridSpacing;
    Vector3<R> _origin;
    ConstArrayAccessor3<T> _accessor;
};

//! Type alias for 3-D cubic array sampler.
template <typename T, typename R> using CubicArraySampler3
    = CubicArraySampler<T, R, 3>;

}  // namespace jet

#include "detail/array_samplers3-inl.h"

#endif  // INCLUDE_JET_ARRAY_SAMPLERS3_H_
