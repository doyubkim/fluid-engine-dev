// Copyright (c) 2018 Doyub Kim
//
// I am making my contributions/submissions to this project solely in my
// personal capacity and am not conveying any rights to any intellectual
// property of any third parties.

#ifdef _MSC_VER
#pragma warning(disable: 4244)
#endif

#include <pch.h>

#include <fbs_helpers.h>
#include <generated/vector_grid2_generated.h>

#include <jet/array_samplers2.h>
#include <jet/vector_grid2.h>

#include <flatbuffers/flatbuffers.h>

#include <algorithm>
#include <string>
#include <vector>

using namespace jet;

VectorGrid2::VectorGrid2() {
}

VectorGrid2::~VectorGrid2() {
}

void VectorGrid2::clear() {
    resize(Size2(), gridSpacing(), origin(), Vector2D());
}

void VectorGrid2::resize(
    size_t resolutionX,
    size_t resolutionY,
    double gridSpacingX,
    double gridSpacingY,
    double originX,
    double originY,
    double initialValueX,
    double initialValueY) {
    resize(
        Size2(resolutionX, resolutionY),
        Vector2D(gridSpacingX, gridSpacingY),
        Vector2D(originX, originY),
        Vector2D(initialValueX, initialValueY));
}

void VectorGrid2::resize(
    const Size2& resolution,
    const Vector2D& gridSpacing,
    const Vector2D& origin,
    const Vector2D& initialValue) {
    setSizeParameters(resolution, gridSpacing, origin);

    onResize(resolution, gridSpacing, origin, initialValue);
}

void VectorGrid2::resize(
    double gridSpacingX,
    double gridSpacingY,
    double originX,
    double originY) {
    resize(
        Vector2D(gridSpacingX, gridSpacingY),
        Vector2D(originX, originY));
}

void VectorGrid2::resize(const Vector2D& gridSpacing, const Vector2D& origin) {
    resize(resolution(), gridSpacing, origin);
}

void VectorGrid2::serialize(std::vector<uint8_t>* buffer) const {
    flatbuffers::FlatBufferBuilder builder(1024);

    auto fbsResolution = jetToFbs(resolution());
    auto fbsGridSpacing = jetToFbs(gridSpacing());
    auto fbsOrigin = jetToFbs(origin());

    std::vector<double> gridData;
    getData(&gridData);
    auto data = builder.CreateVector(gridData.data(), gridData.size());

    auto fbsGrid = fbs::CreateVectorGrid2(
        builder, &fbsResolution, &fbsGridSpacing, &fbsOrigin, data);

    builder.Finish(fbsGrid);

    uint8_t *buf = builder.GetBufferPointer();
    size_t size = builder.GetSize();

    buffer->resize(size);
    memcpy(buffer->data(), buf, size);
}

void VectorGrid2::deserialize(const std::vector<uint8_t>& buffer) {
    auto fbsGrid = fbs::GetVectorGrid2(buffer.data());

    resize(
        fbsToJet(*fbsGrid->resolution()),
        fbsToJet(*fbsGrid->gridSpacing()),
        fbsToJet(*fbsGrid->origin()));

    auto data = fbsGrid->data();
    std::vector<double> gridData(data->size());
    std::copy(data->begin(), data->end(), gridData.begin());

    setData(gridData);
}


VectorGridBuilder2::VectorGridBuilder2() {
}

VectorGridBuilder2::~VectorGridBuilder2() {
}
