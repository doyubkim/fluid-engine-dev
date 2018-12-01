// Copyright (c) 2018 Doyub Kim
//
// I am making my contributions/submissions to this project solely in my
// personal capacity and am not conveying any rights to any intellectual
// property of any third parties.

#include <pch.h>

#include <jet/array_utils.h>
#include <jet/collocated_vector_grid.h>
#include <jet/jet.h>
#include <jet/parallel.h>

namespace jet {

template <size_t N>
CollocatedVectorGrid<N>::CollocatedVectorGrid()
    : _linearSampler(_data, Vector<double, N>::makeConstant(1.0),
                     Vector<double, N>()) {}

template <size_t N>
CollocatedVectorGrid<N>::~CollocatedVectorGrid() {}

template <size_t N>
const Vector<double, N> &CollocatedVectorGrid<N>::operator()(
    const Vector<size_t, N> &idx) const {
    return _data(idx);
}

template <size_t N>
Vector<double, N> &CollocatedVectorGrid<N>::operator()(
    const Vector<size_t, N> &idx) {
    return _data(idx);
}

template <size_t N>
double CollocatedVectorGrid<N>::divergenceAtDataPoint(
    const Vector<size_t, N> &idx) const {
    return GetFdmUtils<N>::divergence(_data, gridSpacing(), idx);
}

template <size_t N>
typename GetCurl<N>::type CollocatedVectorGrid<N>::curlAtDataPoint(
    const Vector<size_t, N> &idx) const {
    return GetFdmUtils<N>::curl(_data, gridSpacing(), idx);
}

template <size_t N>
Vector<double, N> CollocatedVectorGrid<N>::sample(
    const Vector<double, N> &x) const {
    return _sampler(x);
}

template <size_t N>
double CollocatedVectorGrid<N>::divergence(const Vector<double, N> &x) const {
    constexpr size_t kNumPoints = 1 << N;
    std::array<Vector<size_t, N>, kNumPoints> indices;
    std::array<double, kNumPoints> weights;
    _linearSampler.getCoordinatesAndWeights(x, indices, weights);

    double result = 0.0;

    for (int i = 0; i < kNumPoints; ++i) {
        result += weights[i] * divergenceAtDataPoint(indices[i]);
    }

    return result;
}

template <size_t N>
typename GetCurl<N>::type CollocatedVectorGrid<N>::curl(
    const Vector<double, N> &x) const {
    constexpr size_t kNumPoints = 1 << N;
    std::array<Vector<size_t, N>, kNumPoints> indices;
    std::array<double, kNumPoints> weights;
    _linearSampler.getCoordinatesAndWeights(x, indices, weights);

    typename GetCurl<N>::type result{};

    for (int i = 0; i < kNumPoints; ++i) {
        result += weights[i] * curlAtDataPoint(indices[i]);
    }

    return result;
}

template <size_t N>
std::function<Vector<double, N>(const Vector<double, N> &)>
CollocatedVectorGrid<N>::sampler() const {
    return _sampler;
}

template <size_t N>
typename CollocatedVectorGrid<N>::VectorDataView
CollocatedVectorGrid<N>::dataView() {
    return CollocatedVectorGrid<N>::VectorDataView{_data};
}

template <size_t N>
typename CollocatedVectorGrid<N>::ConstVectorDataView
CollocatedVectorGrid<N>::dataView() const {
    return CollocatedVectorGrid<N>::ConstVectorDataView{_data};
}

template <size_t N>
GridDataPositionFunc<N> CollocatedVectorGrid<N>::dataPosition() const {
    Vector<double, N> dataOrigin_ = dataOrigin();
    Vector<double, N> gridSpacing_ = gridSpacing();
    return GridDataPositionFunc<N>(
        [dataOrigin_,
         gridSpacing_](const Vector<size_t, N> &idx) -> Vector<double, N> {
            return dataOrigin_ +
                   elemMul(gridSpacing_, idx.template castTo<double>());
        });
}

template <size_t N>
void CollocatedVectorGrid<N>::forEachDataPointIndex(
    const std::function<void(const Vector<size_t, N> &)> &func) const {
    forEachIndex(_data.size(), GetUnroll<void, N>::unroll(func));
}

template <size_t N>
void CollocatedVectorGrid<N>::parallelForEachDataPointIndex(
    const std::function<void(const Vector<size_t, N> &)> &func) const {
    parallelForEachIndex(_data.size(), GetUnroll<void, N>::unroll(func));
}

template <size_t N>
void CollocatedVectorGrid<N>::swapCollocatedVectorGrid(
    CollocatedVectorGrid *other) {
    swapGrid(other);

    _data.swap(other->_data);
    std::swap(_linearSampler, other->_linearSampler);
    std::swap(_sampler, other->_sampler);
}

template <size_t N>
void CollocatedVectorGrid<N>::setCollocatedVectorGrid(
    const CollocatedVectorGrid &other) {
    setGrid(other);

    _data.copyFrom(other._data);
    resetSampler();
}

template <size_t N>
void CollocatedVectorGrid<N>::onResize(const Vector<size_t, N> &resolution,
                                       const Vector<double, N> &gridSpacing,
                                       const Vector<double, N> &origin,
                                       const Vector<double, N> &initialValue) {
    UNUSED_VARIABLE(resolution);
    UNUSED_VARIABLE(gridSpacing);
    UNUSED_VARIABLE(origin);

    _data.resize(dataSize(), initialValue);
    resetSampler();
}

template <size_t N>
void CollocatedVectorGrid<N>::resetSampler() {
    _linearSampler = LinearArraySampler<Vector<double, N>, N>(
        _data, gridSpacing(), dataOrigin());
    _sampler = _linearSampler.functor();
}

template <size_t N>
void CollocatedVectorGrid<N>::getData(Array1<double> &data) const {
    size_t size = N * product(dataSize(), kOneSize);
    data.resize(size);
    size_t cnt = 0;
    forEachIndex(_data.size(), [&](auto... indices) {
        const Vector<double, N> &value = _data(indices...);
        for (size_t c = 0; c < N; ++c) {
            data[cnt++] = value[c];
        }
    });
}

template <size_t N>
void CollocatedVectorGrid<N>::setData(const ConstArrayView1<double> &data) {
    JET_ASSERT(N * product(dataSize(), kOneSize) == data.length());

    size_t cnt = 0;
    forEachIndex(_data.size(), [&](auto... indices) {
        for (size_t c = 0; c < N; ++c) {
            _data(indices...)[c] = data[cnt++];
        }
    });
}

template class CollocatedVectorGrid<2>;

template class CollocatedVectorGrid<3>;

}  // namespace jet
