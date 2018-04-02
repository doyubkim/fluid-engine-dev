// Copyright (c) 2018 Doyub Kim
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

//! Vertex format enums.
enum class VertexFormat {
    //! Position in 3D.
    Position3 = 1 << 0,

    //! Normal in 3D.
    Normal3 = 1 << 1,

    //! Texture coordinates in 2D.
    TexCoord2 = 1 << 2,

    //! Texture coordinates in 3D.
    TexCoord3 = 1 << 3,

    //! Color in RGBA (4D).
    Color4 = 1 << 4,

    //! Position (3D) and normal (3D).
    Position3Normal3 = Position3 | Normal3,

    //! Position (3D) and texture coordinates (2D).
    Position3TexCoord2 = Position3 | TexCoord2,

    //! Position (3D) and texture coordinates (3D).
    Position3TexCoord3 = Position3 | TexCoord3,

    //! Position (3D) and color in RGBA (4D).
    Position3Color4 = Position3 | Color4,

    //! Position (3D), normal (3D), and texture coordinates (2D).
    Position3Normal3TexCoord2 = Position3Normal3 | TexCoord2,

    //! Position (3D), normal (3D), and color in RGBA (4D).
    Position3Normal3Color4 = Position3Normal3 | Color4,

    //! Position (3D), texture coordinates (2D), and color in RGBA (4D).
    Position3TexCoord2Color4 = Position3TexCoord2 | Color4,

    //! Position (3D), normal (3D), texture coordinates (2D), and color in RGBA
    //! (4D).
    Position3Normal3TexCoord2Color4 = Position3Normal3TexCoord2 | Color4,

    //! Position (3D), normal (3D), texture coordinates (3D), and color in RGBA
    //! (4D).
    Position3Normal3TexCoord3Color4 = Position3Normal3Color4 | TexCoord3,
};

//! Bit-wise operator for two vertex formats.
inline VertexFormat operator&(VertexFormat a, VertexFormat b) {
    return static_cast<VertexFormat>(static_cast<int>(a) & static_cast<int>(b));
}

//! Vertex with position (3D) data only.
struct VertexPosition3 {
    float x, y, z;
};

//! Vertex with position (3D) and normal (3D) data only.
struct VertexPosition3Normal3 {
    float x, y, z;
    float nx, ny, nz;
};

//! Vertex with position (3D) and texture coordinates (2D) data only.
struct VertexPosition3TexCoord2 {
    float x, y, z;
    float u, v;
};

//! Vertex with position (3D) and texture coordinates (3D) data only.
struct VertexPosition3TexCoord3 {
    float x, y, z;
    float u, v, w;
};

//! Vertex with position (3D) and RGBA color (4D) data only.
struct VertexPosition3Color4 {
    float x, y, z;
    float r, g, b, a;
};

//! Vertex with position (3D), normal (3D), and texture coordinates (3D) data
//! only.
struct VertexPosition3Normal3TexCoord2 {
    float x, y, z;
    float nx, ny, nz;
    float u, v;
};

//! Vertex with position (3D), normal (3D), and RGBA color (4D) data only.
struct VertexPosition3Normal3Color4 {
    float x, y, z;
    float nx, ny, nz;
    float r, g, b, a;
};

//! Vertex with position (3D), texture coordinates (2D), and RGBA color (4D)
//! data only.
struct VertexPosition3TexCoord2Color4 {
    float x, y, z;
    float u, v;
    float r, g, b, a;
};

//! Vertex with position (3D), normal (3D), texture coordinates (2D), and RGBA
//! color (4D) data only.
struct VertexPosition3Normal3TexCoord2Color4 {
    float x, y, z;
    float nx, ny, nz;
    float u, v;
    float r, g, b, a;
};

//! Vertex with position (3D), normal (3D), texture coordinates (3D), and RGBA
//! color (4D) data only.
struct VertexPosition3Normal3TexCoord3Color4 {
    float x, y, z;
    float nx, ny, nz;
    float u, v, w;
    float r, g, b, a;
};

//! Collection of vertex helper functions.
class VertexHelper {
 public:
    //! Returns number of floats for a single vertex with given vertex format.
    static size_t getNumberOfFloats(VertexFormat format);

    //! Returns size of a single vertex with given format in bytes.
    static size_t getSizeInBytes(VertexFormat format);
};

}  // namespace viz
}  // namespace jet

#endif  // INCLUDE_JET_VIZ_VERTEX_H_
