// Copyright (c) 2019 Doyub Kim
//
// I am making my contributions/submissions to this project solely in my
// personal capacity and am not conveying any rights to any intellectual
// property of any third parties.

#ifndef INCLUDE_JET_TRIANGLE_MESH_TO_SDF_H_
#define INCLUDE_JET_TRIANGLE_MESH_TO_SDF_H_

#include <jet/scalar_grid3.h>
#include <jet/triangle_mesh3.h>

namespace jet {

//!
//! \brief Generates signed-distance field out of given triangle mesh.
//!
//! This function generates signed-distance field from a triangle mesh. The sign
//! is determined by TriangleMesh3::isInside (negative means inside).
//!
//! \warning Parameter \p exactBand is no longer used and will be deprecated in
//! next release (v2.x).
//!
//! \param[in]      mesh      The mesh.
//! \param[in,out]  sdf       The output signed-distance field.
//! \param[in]      exactBand This parameter is no longer used.
//!
void triangleMeshToSdf(const TriangleMesh3& mesh, ScalarGrid3* sdf,
                       const unsigned int exactBand = 1);

}  // namespace jet

#endif  // INCLUDE_JET_TRIANGLE_MESH_TO_SDF_H_
