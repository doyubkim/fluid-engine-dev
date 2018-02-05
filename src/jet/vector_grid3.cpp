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
#include <generated/vector_grid3_generated.h>

#include <jet/array_samplers3.h>
#include <jet/vector_grid3.h>

#include <flatbuffers/flatbuffers.h>

#include <algorithm>
#include <string>
#include <vector>

using namespace jet;

VectorGrid3::VectorGrid3() {}

VectorGrid3::~VectorGrid3() {}

void VectorGrid3::clear() {
    resize(Size3(), gridSpacing(), origin(), Vector3D());
}

void VectorGrid3::resize(size_t resolutionX, size_t resolutionY,
                         size_t resolutionZ, double gridSpacingX,
                         double gridSpacingY, double gridSpacingZ,
                         double originX, double originY, double originZ,
                         double initialValueX, double initialValueY,
                         double initialValueZ) {
    resize(Size3(resolutionX, resolutionY, resolutionZ),
           Vector3D(gridSpacingX, gridSpacingY, gridSpacingZ),
           Vector3D(originX, originY, originZ),
           Vector3D(initialValueX, initialValueY, initialValueZ));
}

void VectorGrid3::resize(const Size3& resolution, const Vector3D& gridSpacing,
                         const Vector3D& origin, const Vector3D& initialValue) {
    setSizeParameters(resolution, gridSpacing, origin);

    onResize(resolution, gridSpacing, origin, initialValue);
}

void VectorGrid3::resize(double gridSpacingX, double gridSpacingY,
                         double gridSpacingZ, double originX, double originY,
                         double originZ) {
    resize(Vector3D(gridSpacingX, gridSpacingY, gridSpacingZ),
           Vector3D(originX, originY, originZ));
}

void VectorGrid3::resize(const Vector3D& gridSpacing, const Vector3D& origin) {
    resize(resolution(), gridSpacing, origin);
}

void VectorGrid3::serialize(std::vector<uint8_t>* buffer) const {
    flatbuffers::FlatBufferBuilder builder(1024);

    auto fbsResolution = jetToFbs(resolution());
    auto fbsGridSpacing = jetToFbs(gridSpacing());
    auto fbsOrigin = jetToFbs(origin());

    std::vector<double> gridData;
    getData(&gridData);
    auto data = builder.CreateVector(gridData.data(), gridData.size());

    auto fbsGrid = fbs::CreateVectorGrid3(builder, &fbsResolution,
                                          &fbsGridSpacing, &fbsOrigin, data);

    builder.Finish(fbsGrid);

    uint8_t* buf = builder.GetBufferPointer();
    size_t size = builder.GetSize();

    buffer->resize(size);
    memcpy(buffer->data(), buf, size);
}

void VectorGrid3::deserialize(const std::vector<uint8_t>& buffer) {
    auto fbsGrid = fbs::GetVectorGrid3(buffer.data());

    resize(fbsToJet(*fbsGrid->resolution()), fbsToJet(*fbsGrid->gridSpacing()),
           fbsToJet(*fbsGrid->origin()));

    auto data = fbsGrid->data();
    std::vector<double> gridData(data->size());
    std::copy(data->begin(), data->end(), gridData.begin());

    setData(gridData);
}

VectorGridBuilder3::VectorGridBuilder3() {}

VectorGridBuilder3::~VectorGridBuilder3() {}
