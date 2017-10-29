// Copyright (c) 2017 Doyub Kim
//
// I am making my contributions/submissions to this project solely in my
// personal capacity and am not conveying any rights to any intellectual
// property of any third parties.

#ifndef INCLUDE_JET_VIZ_VERTEX_H_
#define INCLUDE_JET_VIZ_VERTEX_H_

#include <jet/vector4.h>
#include <cstdint>

namespace jet {
namespace viz {
enum class VertexFormat {
    Position3 = 1 << 0,

    Normal3 = 1 << 1,
    TexCoord2 = 1 << 2,
    TexCoord3 = 1 << 3,
    Color4 = 1 << 4,

    Position3Normal3 = Position3 | Normal3,
    Position3TexCoord2 = Position3 | TexCoord2,
    Position3TexCoord3 = Position3 | TexCoord3,
    Position3Color4 = Position3 | Color4,

    Position3Normal3TexCoord2 = Position3Normal3 | TexCoord2,
    Position3Normal3Color4 = Position3Normal3 | Color4,
    Position3TexCoord2Color4 = Position3TexCoord2 | Color4,

    Position3Normal3TexCoord2Color4 = Position3Normal3TexCoord2 | Color4,
    Position3Normal3TexCoord3Color4 = Position3Normal3Color4 | TexCoord3,
};

inline VertexFormat operator&(VertexFormat a, VertexFormat b) {
    return static_cast<VertexFormat>(static_cast<int>(a) & static_cast<int>(b));
}

struct VertexPosition3 {
    float x, y, z;
};

struct VertexPosition3Normal3 {
    float x, y, z;
    float nx, ny, nz;
};

struct VertexPosition3TexCoord2 {
    float x, y, z;
    float u, v;
};

struct VertexPosition3TexCoord3 {
    float x, y, z;
    float u, v, w;
};

struct VertexPosition3Color4 {
    float x, y, z;
    float r, g, b, a;
};

struct VertexPosition3Normal3TexCoord2 {
    float x, y, z;
    float nx, ny, nz;
    float u, v;
};

struct VertexPosition3Normal3Color4 {
    float x, y, z;
    float nx, ny, nz;
    float r, g, b, a;
};

struct VertexPosition3TexCoord2Color4 {
    float x, y, z;
    float u, v;
    float r, g, b, a;
};

struct VertexPosition3Normal3TexCoord2Color4 {
    float x, y, z;
    float nx, ny, nz;
    float u, v;
    float r, g, b, a;
};

struct VertexPosition3Normal3TexCoord3Color4 {
    float x, y, z;
    float nx, ny, nz;
    float u, v, w;
    float r, g, b, a;
};

class VertexHelper {
 public:
    static size_t getNumberOfFloats(VertexFormat format);

    static size_t getSizeInBytes(VertexFormat format);
};

}  // namespace viz
}  // namespace jet

#endif  // INCLUDE_JET_VIZ_VERTEX_H_
