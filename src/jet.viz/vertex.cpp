// Copyright (c) 2017 Doyub Kim
//
// I am making my contributions/submissions to this project solely in my
// personal capacity and am not conveying any rights to any intellectual
// property of any third parties.

#include <pch.h>

#include <jet.viz/vertex.h>

using namespace jet;
using namespace viz;

std::size_t VertexHelper::getNumberOfFloats(VertexFormat format) {
    std::size_t size = 0;

    if (static_cast<int>(format & VertexFormat::Position3)) {
        size += 3;
    }

    if (static_cast<int>(format & VertexFormat::Normal3)) {
        size += 3;
    }

    if (static_cast<int>(format & VertexFormat::TexCoord2)) {
        size += 2;
    }

    if (static_cast<int>(format & VertexFormat::TexCoord3)) {
        size += 3;
    }

    if (static_cast<int>(format & VertexFormat::Color4)) {
        size += 4;
    }

    return size;
}

std::size_t VertexHelper::getSizeInBytes(VertexFormat format) {
    return sizeof(float) * getNumberOfFloats(format);
}
