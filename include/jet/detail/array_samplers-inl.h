// Copyright (c) 2018 Doyub Kim
//
// I am making my contributions/submissions to this project solely in my
// personal capacity and am not conveying any rights to any intellectual
// property of any third parties.

#ifndef INCLUDE_JET_DETAIL_ARRAY_SAMPLERS_INL_H_
#define INCLUDE_JET_DETAIL_ARRAY_SAMPLERS_INL_H_

#include <jet/array_samplers.h>

namespace jet {

namespace internal {

template <typename T, size_t N, size_t I>
struct Lerp {
    using ScalarType = typename GetScalarType<T>::value;

    template <typename View, typename... RemainingIndices>
    static auto call(const View& view, Vector<ssize_t, N> i,
                     Vector<ScalarType, N> t, RemainingIndices... indices) {
        using Next = Lerp<T, N, I - 1>;
        return lerp(Next::call(view, i, t, i[I - 1], indices...),
                    Next::call(view, i, t, i[I - 1] + 1, indices...), t[I - 1]);
    }
};

template <typename T, size_t N>
struct Lerp<T, N, 1> {
    using ScalarType = typename GetScalarType<T>::value;

    template <typename View, typename... RemainingIndices>
    static auto call(const View& view, Vector<ssize_t, N> i,
                     Vector<ScalarType, N> t, RemainingIndices... indices) {
        return lerp(view(i[0], indices...), view(i[0] + 1, indices...), t[0]);
    }
};

template <typename T, size_t N, size_t I>
struct Cubic {
    using ScalarType = typename GetScalarType<T>::value;

    template <typename View, typename CubicInterpolationOp,
              typename... RemainingIndices>
    static auto call(const View& view, Vector<ssize_t, N> i,
                     Vector<ScalarType, N> t, CubicInterpolationOp op,
                     RemainingIndices... indices) {
        using Next = Cubic<T, N, I - 1>;
        return op(
            Next::call(view, i, t, op,
                       std::max(i[I - 1] - 1, (ssize_t)view.size()[I - 1] - 1),
                       indices...),
            Next::call(view, i, t, op, i[I - 1], indices...),
            Next::call(view, i, t, op, i[I - 1] + 1, indices...),
            Next::call(view, i, t, op,
                       std::min(i[I - 1] + 2, (ssize_t)view.size()[I - 1] - 1),
                       indices...),
            t[I - 1]);
    }
};

template <typename T, size_t N>
struct Cubic<T, N, 1> {
    using ScalarType = typename GetScalarType<T>::value;

    template <typename View, typename CubicInterpolationOp,
              typename... RemainingIndices>
    static auto call(const View& view, Vector<ssize_t, N> i,
                     Vector<ScalarType, N> t, CubicInterpolationOp op,
                     RemainingIndices... indices) {
        return op(
            view(std::max(i[0] - 1, (ssize_t)view.size()[0] - 1), indices...),
            view(i[0], indices...), view(i[0] + 1, indices...),
            view(std::min(i[0] + 2, (ssize_t)view.size()[0] - 1), indices...),
            t[0]);
    }
};

template <typename T, size_t N, size_t I>
struct GetCoordinatesAndWeights {
    using ScalarType = typename GetScalarType<T>::value;

    template <typename Coords, typename Weights, typename... RemainingIndices>
    static void call(Coords& c, Weights& w, Vector<size_t, N> i,
                     Vector<ScalarType, N> t, T acc, RemainingIndices... idx) {
        using Next = GetCoordinatesAndWeights<T, N, I - 1>;
        Next::call(c, w, i, t, acc * (1 - t[I - 1]), 0, idx...);
        Next::call(c, w, i, t, acc * t[I - 1], 1, idx...);
    }
};

template <typename T, size_t N>
struct GetCoordinatesAndWeights<T, N, 1> {
    using ScalarType = typename GetScalarType<T>::value;

    template <typename Coords, typename Weights, typename... RemainingIndices>
    static void call(Coords& c, Weights& w, Vector<size_t, N> i,
                     Vector<ScalarType, N> t, T acc, RemainingIndices... idx) {
        c(0, idx...) = Vector<size_t, N>(0, idx...) + i;
        c(1, idx...) = Vector<size_t, N>(1, idx...) + i;

        w(0, idx...) = acc * (1 - t[0]);
        w(1, idx...) = acc * (t[0]);
    }
};

template <typename T, size_t N, size_t I>
struct GetCoordinatesAndGradientWeights {
    template <typename Coords, typename Weights, typename... RemainingIndices>
    static void call(Coords& c, Weights& w, Vector<size_t, N> i, Vector<T, N> t,
                     Vector<T, N> acc, RemainingIndices... idx) {
        Vector<T, N> w0 = Vector<T, N>::makeConstant(1 - t[I - 1]);
        w0[I - 1] = -1;
        Vector<T, N> w1 = Vector<T, N>::makeConstant(t[I - 1]);
        w1[I - 1] = 1;

        using Next = GetCoordinatesAndGradientWeights<T, N, I - 1>;
        Next::call(c, w, i, t, elemMul(acc, w0), 0, idx...);
        Next::call(c, w, i, t, elemMul(acc, w1), 1, idx...);
    }
};

template <typename T, size_t N>
struct GetCoordinatesAndGradientWeights<T, N, 1> {
    template <typename Coords, typename Weights, typename... RemainingIndices>
    static void call(Coords& c, Weights& w, Vector<size_t, N> i, Vector<T, N> t,
                     Vector<T, N> acc, RemainingIndices... idx) {
        c(0, idx...) = Vector<size_t, N>(0, idx...) + i;
        c(1, idx...) = Vector<size_t, N>(1, idx...) + i;

        Vector<T, N> w0 = Vector<T, N>::makeConstant(1 - t[0]);
        w0[0] = -1;
        Vector<T, N> w1 = Vector<T, N>::makeConstant(t[0]);
        w1[0] = 1;

        w(0, idx...) = elemMul(acc, w0);
        w(1, idx...) = elemMul(acc, w1);
    }
};

}  // namespace internal

////////////////////////////////////////////////////////////////////////////////
// MARK: NearestArraySampler

template <typename T, size_t N>
NearestArraySampler<T, N>::NearestArraySampler(
    const ArrayView<const T, N>& view, const VectorType& gridSpacing,
    const VectorType& gridOrigin)
    : _view(view),
      _gridSpacing(gridSpacing),
      _invGridSpacing(ScalarType{1} / gridSpacing),
      _gridOrigin(gridOrigin) {}

template <typename T, size_t N>
NearestArraySampler<T, N>::NearestArraySampler(const NearestArraySampler& other)
    : _view(other._view),
      _gridSpacing(other._gridSpacing),
      _invGridSpacing(other._invGridSpacing),
      _gridOrigin(other._gridOrigin) {}

template <typename T, size_t N>
T NearestArraySampler<T, N>::operator()(const VectorType& pt) const {
    return _view(getCoordinate(pt));
}

template <typename T, size_t N>
typename NearestArraySampler<T, N>::CoordIndexType
NearestArraySampler<T, N>::getCoordinate(const VectorType& pt) const {
    Vector<ssize_t, N> is;
    Vector<ScalarType, N> ts;
    VectorType npt = elemMul(pt - _gridOrigin, _invGridSpacing);
    Vector<ssize_t, N> size = _view.size().template castTo<ssize_t>();

    for (size_t i = 0; i < N; ++i) {
        getBarycentric(npt[i], 0, size[i], is[i], ts[i]);
        is[i] =
            std::min(static_cast<ssize_t>(is[i] + ts[i] + 0.5), size[i] - 1);
    }

    return is.template castTo<size_t>();
}

template <typename T, size_t N>
std::function<T(const typename NearestArraySampler<T, N>::VectorType&)>
NearestArraySampler<T, N>::functor() const {
    NearestArraySampler sampler(*this);
    return [sampler](const VectorType& x) -> T { return sampler(x); };
}

////////////////////////////////////////////////////////////////////////////////
// MARK: LinearArraySampler

template <typename T, size_t N>
LinearArraySampler<T, N>::LinearArraySampler(const ArrayView<const T, N>& view,
                                             const VectorType& gridSpacing,
                                             const VectorType& gridOrigin)
    : _view(view),
      _gridSpacing(gridSpacing),
      _invGridSpacing(ScalarType{1} / gridSpacing),
      _gridOrigin(gridOrigin) {}

template <typename T, size_t N>
LinearArraySampler<T, N>::LinearArraySampler(const LinearArraySampler& other)
    : _view(other._view),
      _gridSpacing(other._gridSpacing),
      _invGridSpacing(other._invGridSpacing),
      _gridOrigin(other._gridOrigin) {}

template <typename T, size_t N>
T LinearArraySampler<T, N>::operator()(const VectorType& pt) const {
    Vector<ssize_t, N> is;
    Vector<ScalarType, N> ts;
    VectorType npt = elemMul(pt - _gridOrigin, _invGridSpacing);
    Vector<ssize_t, N> size = _view.size().template castTo<ssize_t>();

    for (size_t i = 0; i < N; ++i) {
        getBarycentric(npt[i], 0, size[i], is[i], ts[i]);
    }

    return internal::Lerp<T, N, N>::call(_view, is, ts);
}

template <typename T, size_t N>
void LinearArraySampler<T, N>::getCoordinatesAndWeights(
    const VectorType& pt, std::array<CoordIndexType, kFlatKernelSize>& indices,
    std::array<ScalarType, kFlatKernelSize>& weights) const {
    Vector<ssize_t, N> is;
    Vector<ScalarType, N> ts;
    VectorType npt = elemMul(pt - _gridOrigin, _invGridSpacing);
    Vector<ssize_t, N> size = _view.size().template castTo<ssize_t>();

    for (size_t i = 0; i < N; ++i) {
        getBarycentric(npt[i], 0, size[i], is[i], ts[i]);
    }

    Vector<size_t, N> viewSize = Vector<size_t, N>::makeConstant(2);
    ArrayView<CoordIndexType, N> indexView(indices.data(), viewSize);
    ArrayView<ScalarType, N> weightView(weights.data(), viewSize);

    internal::GetCoordinatesAndWeights<ScalarType, N, N>::call(
        indexView, weightView, is.template castTo<size_t>(), ts, 1);
}

template <typename T, size_t N>
void LinearArraySampler<T, N>::getCoordinatesAndGradientWeights(
    const VectorType& pt, std::array<CoordIndexType, kFlatKernelSize>& indices,
    std::array<VectorType, kFlatKernelSize>& weights) const {
    Vector<ssize_t, N> is;
    Vector<ScalarType, N> ts;
    VectorType npt = elemMul(pt - _gridOrigin, _invGridSpacing);
    Vector<ssize_t, N> size = _view.size().template castTo<ssize_t>();

    for (size_t i = 0; i < N; ++i) {
        getBarycentric(npt[i], 0, size[i], is[i], ts[i]);
    }

    Vector<size_t, N> viewSize = Vector<size_t, N>::makeConstant(2);
    ArrayView<CoordIndexType, N> indexView(indices.data(), viewSize);
    ArrayView<VectorType, N> weightView(weights.data(), viewSize);

    internal::GetCoordinatesAndGradientWeights<ScalarType, N, N>::call(
        indexView, weightView, is.template castTo<size_t>(), ts,
        _invGridSpacing);
}

template <typename T, size_t N>
std::function<T(const typename LinearArraySampler<T, N>::VectorType&)>
LinearArraySampler<T, N>::functor() const {
    LinearArraySampler sampler(*this);
    return [sampler](const VectorType& x) -> T { return sampler(x); };
}

////////////////////////////////////////////////////////////////////////////////
// MARK: CubicArraySampler

template <typename T, size_t N, typename CIOp>
CubicArraySampler<T, N, CIOp>::CubicArraySampler(
    const ArrayView<const T, N>& view, const VectorType& gridSpacing,
    const VectorType& gridOrigin)
    : _view(view),
      _gridSpacing(gridSpacing),
      _invGridSpacing(ScalarType{1} / gridSpacing),
      _gridOrigin(gridOrigin) {}

template <typename T, size_t N, typename CIOp>
CubicArraySampler<T, N, CIOp>::CubicArraySampler(const CubicArraySampler& other)
    : _view(other._view),
      _gridSpacing(other._gridSpacing),
      _invGridSpacing(other._invGridSpacing),
      _gridOrigin(other._gridOrigin) {}

template <typename T, size_t N, typename CIOp>
T CubicArraySampler<T, N, CIOp>::operator()(const VectorType& pt) const {
    Vector<ssize_t, N> is;
    Vector<ScalarType, N> ts;
    VectorType npt = elemMul(pt - _gridOrigin, _invGridSpacing);
    Vector<ssize_t, N> size = _view.size().template castTo<ssize_t>();

    for (size_t i = 0; i < N; ++i) {
        getBarycentric(npt[i], 0, size[i], is[i], ts[i]);
    }

    return internal::Cubic<T, N, N>::call(_view, is, ts, CIOp());
}

template <typename T, size_t N, typename CIOp>
std::function<T(const typename CubicArraySampler<T, N, CIOp>::VectorType&)>
CubicArraySampler<T, N, CIOp>::functor() const {
    CubicArraySampler sampler(*this);
    return [sampler](const VectorType& x) -> T { return sampler(x); };
}

}  // namespace jet

#endif  // INCLUDE_JET_DETAIL_ARRAY_SAMPLERS_INL_H_
