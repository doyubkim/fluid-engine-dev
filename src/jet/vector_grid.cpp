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
#include <generated/vector_grid2_generated.h>
#include <generated/vector_grid3_generated.h>

#include <jet/array_samplers.h>
#include <jet/vector_grid.h>

#include <flatbuffers/flatbuffers.h>

namespace jet {

template <size_t N>
struct GetFlatbuffersVectorGrid {};

template <>
struct GetFlatbuffersVectorGrid<2> {
    static flatbuffers::Offset<fbs::VectorGrid2> createVectorGrid(
        flatbuffers::FlatBufferBuilder &_fbb,
        const jet::fbs::Vector2UZ *resolution,
        const jet::fbs::Vector2D *gridSpacing, const jet::fbs::Vector2D *origin,
        flatbuffers::Offset<flatbuffers::Vector<double>> data) {
        return fbs::CreateVectorGrid2(_fbb, resolution, gridSpacing, origin,
                                      data);
    }

    static const jet::fbs::VectorGrid2 *getVectorGrid(const void *buf) {
        return fbs::GetVectorGrid2(buf);
    }
};

template <>
struct GetFlatbuffersVectorGrid<3> {
    static flatbuffers::Offset<fbs::VectorGrid3> createVectorGrid(
        flatbuffers::FlatBufferBuilder &_fbb,
        const jet::fbs::Vector3UZ *resolution,
        const jet::fbs::Vector3D *gridSpacing, const jet::fbs::Vector3D *origin,
        flatbuffers::Offset<flatbuffers::Vector<double>> data) {
        return fbs::CreateVectorGrid3(_fbb, resolution, gridSpacing, origin,
                                      data);
    }

    static const jet::fbs::VectorGrid3 *getVectorGrid(const void *buf) {
        return fbs::GetVectorGrid3(buf);
    }
};

template <size_t N>
VectorGrid<N>::VectorGrid() {}

template <size_t N>
VectorGrid<N>::~VectorGrid() {}

template <size_t N>
void VectorGrid<N>::clear() {
    resize(Vector<size_t, N>(), gridSpacing(), origin(), Vector<double, N>());
}

template <size_t N>
void VectorGrid<N>::resize(const Vector<size_t, N> &resolution,
                           const Vector<double, N> &gridSpacing,
                           const Vector<double, N> &origin,
                           const Vector<double, N> &initialValue) {
    setSizeParameters(resolution, gridSpacing, origin);

    onResize(resolution, gridSpacing, origin, initialValue);
}

template <size_t N>
void VectorGrid<N>::resize(const Vector<double, N> &gridSpacing,
                           const Vector<double, N> &origin) {
    resize(resolution(), gridSpacing, origin);
}

template <size_t N>
void VectorGrid<N>::serialize(std::vector<uint8_t> *buffer) const {
    flatbuffers::FlatBufferBuilder builder(1024);

    auto fbsResolution = jetToFbs(resolution());
    auto fbsGridSpacing = jetToFbs(gridSpacing());
    auto fbsOrigin = jetToFbs(origin());

    Array1<double> gridData;
    getData(gridData);
    auto data = builder.CreateVector(gridData.data(), gridData.length());

    auto fbsGrid = GetFlatbuffersVectorGrid<N>::createVectorGrid(
        builder, &fbsResolution, &fbsGridSpacing, &fbsOrigin, data);

    builder.Finish(fbsGrid);

    uint8_t *buf = builder.GetBufferPointer();
    size_t size = builder.GetSize();

    buffer->resize(size);
    memcpy(buffer->data(), buf, size);
}

template <size_t N>
void VectorGrid<N>::deserialize(const std::vector<uint8_t> &buffer) {
    auto fbsGrid = GetFlatbuffersVectorGrid<N>::getVectorGrid(buffer.data());

    resize(fbsToJet(*fbsGrid->resolution()), fbsToJet(*fbsGrid->gridSpacing()),
           fbsToJet(*fbsGrid->origin()));

    auto data = fbsGrid->data();
    Array1<double> gridData(data->size());
    std::copy(data->begin(), data->end(), gridData.begin());

    setData(gridData);
}

template class VectorGrid<2>;

template class VectorGrid<3>;

template <size_t N>
VectorGridBuilder<N>::VectorGridBuilder() {}

template <size_t N>
VectorGridBuilder<N>::~VectorGridBuilder() {}

template class VectorGridBuilder<2>;

template class VectorGridBuilder<3>;

}  // namespace jet
