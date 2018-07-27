// Copyright (c) 2018 Doyub Kim
//
// I am making my contributions/submissions to this project solely in my
// personal capacity and am not conveying any rights to any intellectual
// property of any third parties.

#ifndef INCLUDE_JET_ARRAY_SAMPLERS_H_
#define INCLUDE_JET_ARRAY_SAMPLERS_H_

#include <jet/array_view.h>
#include <jet/math_utils.h>
#include <jet/matrix.h>

#include <functional>

namespace jet {

////////////////////////////////////////////////////////////////////////////////
// MARK: NearestArraySampler

//!
//! \brief N-D nearest array sampler class.
//!
//! This class provides nearest sampling interface for a given N-D array.
//!
//! \tparam T - The value type to sample.
//! \tparam N - Dimension.
//!
template <typename T, size_t N>
class NearestArraySampler final {
 public:
    static_assert(N > 0, "Dimension should be greater than 0");

    using ScalarType = typename GetScalarType<T>::value;

    static_assert(std::is_floating_point<ScalarType>::value,
                  "NearestArraySampler only can be instantiated with floating "
                  "point types");

    using VectorType = Vector<ScalarType, N>;
    using CoordIndexType = Vector<size_t, N>;

    //!
    //! \brief      Constructs a sampler.
    //!
    //! \param[in]  view        The array view.
    //! \param[in]  gridSpacing The grid spacing.
    //! \param[in]  gridOrigin  The grid origin.
    //!
    explicit NearestArraySampler(const ArrayView<const T, N>& view,
                                 const VectorType& gridSpacing,
                                 const VectorType& gridOrigin);

    //! Copy constructor.
    NearestArraySampler(const NearestArraySampler& other);

    //! Returns sampled value at point \p pt.
    T operator()(const VectorType& pt) const;

    //! Returns the nearest array index for point \p x.
    CoordIndexType getCoordinate(const VectorType& pt) const;

    //! Returns a funtion object that wraps this instance.
    std::function<T(const VectorType&)> functor() const;

 private:
    ArrayView<const T, N> _view;
    VectorType _gridSpacing;
    VectorType _invGridSpacing;
    VectorType _gridOrigin;
};

template <typename T>
using NearestArraySampler1 = NearestArraySampler<T, 1>;

template <typename T>
using NearestArraySampler2 = NearestArraySampler<T, 2>;

template <typename T>
using NearestArraySampler3 = NearestArraySampler<T, 3>;

////////////////////////////////////////////////////////////////////////////////
// MARK: LinearArraySampler

//!
//! \brief N-D array sampler using linear interpolation.
//!
//! This class provides linear sampling interface for a given N-D array.
//!
//! \tparam T - The value type to sample.
//! \tparam N - Dimension.
//!
template <typename T, size_t N>
class LinearArraySampler final {
 public:
    static_assert(N > 0, "N should be greater than 0");

    using ScalarType = typename GetScalarType<T>::value;

    static_assert(std::is_floating_point<ScalarType>::value,
                  "LinearArraySampler only can be instantiated with floating "
                  "point types");

    using VectorType = Vector<ScalarType, N>;
    using CoordIndexType = Vector<size_t, N>;

    static constexpr size_t kFlatKernelSize = 1 << N;

    //!
    //! \brief      Constructs a sampler.
    //!
    //! \param[in]  view        The array view.
    //! \param[in]  gridSpacing The grid spacing.
    //! \param[in]  gridOrigin  The grid origin.
    //!
    explicit LinearArraySampler(const ArrayView<const T, N>& view,
                                const VectorType& gridSpacing,
                                const VectorType& gridOrigin);

    //! Copy constructor.
    LinearArraySampler(const LinearArraySampler& other);

    //! Returns sampled value at point \p pt.
    T operator()(const VectorType& pt) const;

    //! Returns the indices of points and their sampling weight for given point.
    void getCoordinatesAndWeights(
        const VectorType& pt,
        std::array<CoordIndexType, kFlatKernelSize>& indices,
        std::array<ScalarType, kFlatKernelSize>& weights) const;

    //! Returns the indices of points and their gradient of sampling weight for
    //! given point.
    void getCoordinatesAndGradientWeights(
        const VectorType& pt,
        std::array<CoordIndexType, kFlatKernelSize>& indices,
        std::array<VectorType, kFlatKernelSize>& weights) const;

    //! Returns a std::function instance that wraps this instance.
    std::function<T(const VectorType&)> functor() const;

 private:
    ArrayView<const T, N> _view;
    VectorType _gridSpacing;
    VectorType _invGridSpacing;
    VectorType _gridOrigin;
};

template <typename T>
using LinearArraySampler1 = LinearArraySampler<T, 1>;

template <typename T>
using LinearArraySampler2 = LinearArraySampler<T, 2>;

template <typename T>
using LinearArraySampler3 = LinearArraySampler<T, 3>;

////////////////////////////////////////////////////////////////////////////////
// MARK: CubicArraySampler

//!
//! \brief N-D cubic array sampler class.
//!
//! This class provides cubic sampling interface for a given N-D array.
//!
//! \tparam T - The value type to sample.
//! \tparam N - Dimension.
//!
template <typename T, size_t N, typename CubicInterpolationOp>
class CubicArraySampler final {
 public:
    static_assert(N > 0, "N should be greater than 0");

    using ScalarType = typename GetScalarType<T>::value;

    static_assert(std::is_floating_point<ScalarType>::value,
                  "CubicArraySampler only can be instantiated with floating "
                  "point types");

    using VectorType = Vector<ScalarType, N>;
    using CoordIndexType = Vector<size_t, N>;

    //!
    //! \brief      Constructs a sampler.
    //!
    //! \param[in]  view        The array view.
    //! \param[in]  gridSpacing The grid spacing.
    //! \param[in]  gridOrigin  The grid origin.
    //!
    explicit CubicArraySampler(const ArrayView<const T, N>& view,
                               const VectorType& gridSpacing,
                               const VectorType& gridOrigin);

    //! Copy constructor.
    CubicArraySampler(const CubicArraySampler& other);

    //! Returns sampled value at point \p pt.
    T operator()(const VectorType& pt) const;

    //! Returns a funtion object that wraps this instance.
    std::function<T(const VectorType&)> functor() const;

 private:
    ArrayView<const T, N> _view;
    VectorType _gridSpacing;
    VectorType _invGridSpacing;
    VectorType _gridOrigin;
};

template <typename T>
struct CatmullRom {
    using ScalarType = typename GetScalarType<T>::value;

    T operator()(const T& f0, const T& f1, const T& f2, const T& f3,
                 ScalarType t) const {
        return catmullRom(f0, f1, f2, f3, t);
    }
};

template <typename T>
struct MonotonicCatmullRom {
    using ScalarType = typename GetScalarType<T>::value;

    T operator()(const T& f0, const T& f1, const T& f2, const T& f3,
                 ScalarType t) const {
        return monotonicCatmullRom(f0, f1, f2, f3, t);
    }
};

template <typename T>
using CatmullRomArraySampler1 =
    CubicArraySampler<T, 1, CatmullRom<T>>;

template <typename T>
using CatmullRomArraySampler2 =
    CubicArraySampler<T, 2, CatmullRom<T>>;

template <typename T>
using CatmullRomArraySampler3 =
    CubicArraySampler<T, 3, CatmullRom<T>>;

template <typename T>
using MonotonicCatmullRomArraySampler1 =
    CubicArraySampler<T, 1, MonotonicCatmullRom<T>>;

template <typename T>
using MonotonicCatmullRomArraySampler2 =
    CubicArraySampler<T, 2, MonotonicCatmullRom<T>>;

template <typename T>
using MonotonicCatmullRomArraySampler3 =
    CubicArraySampler<T, 3, MonotonicCatmullRom<T>>;

}  // namespace jet

#include <jet/detail/array_samplers-inl.h>

#endif  // INCLUDE_JET_ARRAY_SAMPLERS_H_
