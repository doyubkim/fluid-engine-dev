// Copyright (c) 2018 Doyub Kim
//
// I am making my contributions/submissions to this project solely in my
// personal capacity and am not conveying any rights to any intellectual
// property of any third parties.

#ifdef _MSC_VER
#pragma warning(disable : 4244)
#endif

#include <pch.h>

#include <fbs_helpers.h>
#include <generated/scalar_grid2_generated.h>
#include <generated/scalar_grid3_generated.h>

#include <jet/fdm_utils.h>
#include <jet/parallel.h>
#include <jet/scalar_grid.h>

#include <flatbuffers/flatbuffers.h>

namespace jet {

// MARK: Serialization helpers

template <size_t N>
struct GetFlatbuffersScalarGrid {};

template <>
struct GetFlatbuffersScalarGrid<2> {
    static flatbuffers::Offset<fbs::ScalarGrid2> createScalarGrid(
        flatbuffers::FlatBufferBuilder &_fbb,
        const jet::fbs::Vector2UZ *resolution,
        const jet::fbs::Vector2D *gridSpacing, const jet::fbs::Vector2D *origin,
        flatbuffers::Offset<flatbuffers::Vector<double>> data) {
        return fbs::CreateScalarGrid2(_fbb, resolution, gridSpacing, origin,
                                      data);
    }

    static const jet::fbs::ScalarGrid2 *getScalarGrid(const void *buf) {
        return fbs::GetScalarGrid2(buf);
    }
};

template <>
struct GetFlatbuffersScalarGrid<3> {
    static flatbuffers::Offset<fbs::ScalarGrid3> createScalarGrid(
        flatbuffers::FlatBufferBuilder &_fbb,
        const jet::fbs::Vector3UZ *resolution,
        const jet::fbs::Vector3D *gridSpacing, const jet::fbs::Vector3D *origin,
        flatbuffers::Offset<flatbuffers::Vector<double>> data) {
        return fbs::CreateScalarGrid3(_fbb, resolution, gridSpacing, origin,
                                      data);
    }

    static const jet::fbs::ScalarGrid3 *getScalarGrid(const void *buf) {
        return fbs::GetScalarGrid3(buf);
    }
};

// MARK: ScalarGrid implementations

template <size_t N>
ScalarGrid<N>::ScalarGrid()
    : _linearSampler(LinearArraySampler<double, N>(
          _data, Vector<double, N>::makeConstant(1), Vector<double, N>())) {}

template <size_t N>
ScalarGrid<N>::~ScalarGrid() {}

template <size_t N>
void ScalarGrid<N>::clear() {
    resize(Vector<size_t, N>(), gridSpacing(), origin(), 0.0);
}

template <size_t N>
void ScalarGrid<N>::resize(const Vector<size_t, N> &resolution,
                           const Vector<double, N> &gridSpacing,
                           const Vector<double, N> &origin,
                           double initialValue) {
    setSizeParameters(resolution, gridSpacing, origin);

    _data.resize(dataSize(), initialValue);
    resetSampler();
}

template <size_t N>
void ScalarGrid<N>::resize(const Vector<double, N> &gridSpacing,
                           const Vector<double, N> &origin) {
    resize(resolution(), gridSpacing, origin);
}

template <size_t N>
const double &ScalarGrid<N>::operator()(const Vector<size_t, N> &idx) const {
    return _data(idx);
}

template <size_t N>
double &ScalarGrid<N>::operator()(const Vector<size_t, N> &idx) {
    return _data(idx);
}

template <size_t N>
Vector<double, N> ScalarGrid<N>::gradientAtDataPoint(
    const Vector<size_t, N> &idx) const {
    return GetFdmUtils<N>::gradient(_data, gridSpacing(), idx);
}

template <size_t N>
double ScalarGrid<N>::laplacianAtDataPoint(const Vector<size_t, N> &idx) const {
    return GetFdmUtils<N>::laplacian(_data, gridSpacing(), idx);
}

template <size_t N>
double ScalarGrid<N>::sample(const Vector<double, N> &x) const {
    return _sampler(x);
}

template <size_t N>
std::function<double(const Vector<double, N> &)> ScalarGrid<N>::sampler()
    const {
    return _sampler;
}

template <size_t N>
Vector<double, N> ScalarGrid<N>::gradient(const Vector<double, N> &x) const {
    constexpr size_t kNumPoints = 1u << N;
    std::array<Vector<size_t, N>, kNumPoints> indices;
    std::array<double, kNumPoints> weights;
    _linearSampler.getCoordinatesAndWeights(x, indices, weights);

    Vector<double, N> result;

    for (size_t i = 0; i < kNumPoints; ++i) {
        result += weights[i] * gradientAtDataPoint(indices[i]);
    }

    return result;
}

template <size_t N>
double ScalarGrid<N>::laplacian(const Vector<double, N> &x) const {
    constexpr size_t kNumPoints = 1u << N;
    std::array<Vector<size_t, N>, kNumPoints> indices;
    std::array<double, kNumPoints> weights;
    _linearSampler.getCoordinatesAndWeights(x, indices, weights);

    double result = 0.0;

    for (size_t i = 0; i < kNumPoints; ++i) {
        result += weights[i] * laplacianAtDataPoint(indices[i]);
    }

    return result;
}

template <size_t N>
typename ScalarGrid<N>::ScalarDataView ScalarGrid<N>::dataView() {
    return ScalarDataView(_data);
}

template <size_t N>
typename ScalarGrid<N>::ConstScalarDataView ScalarGrid<N>::dataView() const {
    return ConstScalarDataView(_data);
}

template <size_t N>
GridDataPositionFunc<N> ScalarGrid<N>::dataPosition() const {
    Vector<double, N> o = dataOrigin();
    Vector<double, N> gs = gridSpacing();
    return GridDataPositionFunc<N>(
        [o, gs](const Vector<size_t, N> &idx) -> Vector<double, N> {
            return o + elemMul(gs, idx.template castTo<double>());
        });
}

template <size_t N>
void ScalarGrid<N>::fill(double value, ExecutionPolicy policy) {
    parallelForEachIndex(
        Vector<size_t, N>(), _data.size(),
        [this, value](auto... indices) { _data(indices...) = value; }, policy);
}

template <size_t N>
void ScalarGrid<N>::fill(
    const std::function<double(const Vector<double, N> &)> &func,
    ExecutionPolicy policy) {
    auto pos = dataPosition();
    parallelForEachIndex(Vector<size_t, N>(), _data.size(),
                         [this, &func, &pos](auto... indices) {
                             _data(indices...) =
                                 func(pos(Vector<size_t, N>(indices...)));
                         },
                         policy);
}

template <size_t N>
void ScalarGrid<N>::forEachDataPointIndex(
    const std::function<void(const Vector<size_t, N> &)> &func) const {
    forEachIndex(_data.size(), GetUnroll<void, N>::unroll(func));
}

template <size_t N>
void ScalarGrid<N>::parallelForEachDataPointIndex(
    const std::function<void(const Vector<size_t, N> &)> &func) const {
    parallelForEachIndex(_data.size(), GetUnroll<void, N>::unroll(func));
}

template <size_t N>
void ScalarGrid<N>::serialize(std::vector<uint8_t> *buffer) const {
    flatbuffers::FlatBufferBuilder builder(1024);

    auto fbsResolution = jetToFbs(resolution());
    auto fbsGridSpacing = jetToFbs(gridSpacing());
    auto fbsOrigin = jetToFbs(origin());

    Array1<double> gridData;
    getData(gridData);
    auto data = builder.CreateVector(gridData.data(), gridData.length());

    auto fbsGrid = GetFlatbuffersScalarGrid<N>::createScalarGrid(
        builder, &fbsResolution, &fbsGridSpacing, &fbsOrigin, data);

    builder.Finish(fbsGrid);

    uint8_t *buf = builder.GetBufferPointer();
    size_t size = builder.GetSize();

    buffer->resize(size);
    memcpy(buffer->data(), buf, size);
}

template <size_t N>
void ScalarGrid<N>::deserialize(const std::vector<uint8_t> &buffer) {
    auto fbsGrid = GetFlatbuffersScalarGrid<N>::getScalarGrid(buffer.data());

    resize(fbsToJet(*fbsGrid->resolution()), fbsToJet(*fbsGrid->gridSpacing()),
           fbsToJet(*fbsGrid->origin()));

    auto data = fbsGrid->data();
    Array1<double> gridData(data->size());
    std::copy(data->begin(), data->end(), gridData.begin());

    setData(gridData);
}

template <size_t N>
void ScalarGrid<N>::swapScalarGrid(ScalarGrid *other) {
    swapGrid(other);

    _data.swap(other->_data);
    std::swap(_linearSampler, other->_linearSampler);
    std::swap(_sampler, other->_sampler);
}

template <size_t N>
void ScalarGrid<N>::setScalarGrid(const ScalarGrid &other) {
    setGrid(other);

    _data.copyFrom(other._data);
    resetSampler();
}

template <size_t N>
void ScalarGrid<N>::resetSampler() {
    _linearSampler =
        LinearArraySampler<double, N>(_data, gridSpacing(), dataOrigin());
    _sampler = _linearSampler.functor();
}

template <size_t N>
void ScalarGrid<N>::getData(Array1<double> &data) const {
    size_t size = product(dataSize(), kOneSize);
    data.resize(size);
    std::copy(_data.begin(), _data.end(), data.begin());
}

template <size_t N>
void ScalarGrid<N>::setData(const ConstArrayView1<double> &data) {
    JET_ASSERT(product(dataSize(), kOneSize) == data.length());

    std::copy(data.begin(), data.end(), _data.begin());
}

template class ScalarGrid<2>;

template class ScalarGrid<3>;

// MARK: ScalarGridBuilder implementations

template <size_t N>
ScalarGridBuilder<N>::ScalarGridBuilder() {}

template <size_t N>
ScalarGridBuilder<N>::~ScalarGridBuilder() {}

template class ScalarGridBuilder<2>;

template class ScalarGridBuilder<3>;

}  // namespace jet
