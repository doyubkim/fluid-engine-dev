// Copyright (c) 2017 Doyub Kim
//
// I am making my contributions/submissions to this project solely in my
// personal capacity and am not conveying any rights to any intellectual
// property of any third parties.

#ifndef INCLUDE_JET_VIZ_PRIMITIVE_TYPES_H_
#define INCLUDE_JET_VIZ_PRIMITIVE_TYPES_H_

#include "shader.h"
#include "vertex.h"

namespace jet { namespace viz {

enum class PrimitiveType {
    Points = 0,
    Lines,
    LineStrip,
    Triangles,
    TriangleStrip,
};
} }  // namespace jet::viz

#endif  // INCLUDE_JET_VIZ_PRIMITIVE_TYPES_H_
