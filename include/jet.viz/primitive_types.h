// Copyright (c) 2018 Doyub Kim
//
// I am making my contributions/submissions to this project solely in my
// personal capacity and am not conveying any rights to any intellectual
// property of any third parties.

#ifndef INCLUDE_JET_VIZ_PRIMITIVE_TYPES_H_
#define INCLUDE_JET_VIZ_PRIMITIVE_TYPES_H_

#include "shader.h"
#include "vertex.h"

namespace jet {

namespace viz {

//! Render primitive types
enum class PrimitiveType {
    //! Multiple points primitive.
    Points = 0,

    //! Multiple lines primitive.
    Lines,

    //! Line strip primitive.
    LineStrip,

    //! Multiple triangles primitive.
    Triangles,

    //! Triangle strip primitive.
    TriangleStrip,
};

}  // namespace viz

}  // namespace jet

#endif  // INCLUDE_JET_VIZ_PRIMITIVE_TYPES_H_
