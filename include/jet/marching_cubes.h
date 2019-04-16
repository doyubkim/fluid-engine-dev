// Copyright (c) 2019 Doyub Kim
//
// I am making my contributions/submissions to this project solely in my
// personal capacity and am not conveying any rights to any intellectual
// property of any third parties.

#ifndef INCLUDE_JET_MARCHING_CUBES_H_
#define INCLUDE_JET_MARCHING_CUBES_H_

#include <jet/array_accessor3.h>
#include <jet/constants.h>
#include <jet/triangle_mesh3.h>

namespace jet {

//!
//! \brief      Computes marching cubes and extract triangle mesh from grid.
//!
//! This function computes the marching cube algorithm to extract triangle mesh
//! from the scalar grid field. The triangle mesh will be the iso surface, and
//! the iso value can be specified. For the boundaries (the walls), it can be
//! specified whether to close or open with \p bndClose (default: close all).
//! Another boundary flag \p bndConnectivity can be used for specifying
//! topological connectivity of the boundary meshes (default: disconnect all).
//!
//! \param[in]  grid            The grid.
//! \param[in]  gridSize        The grid size.
//! \param[in]  origin          The origin.
//! \param[out] mesh            The output triangle mesh.
//! \param[in]  isoValue        The iso-surface value.
//! \param[in]  bndClose        The boundary open flag.
//! \param[in]  bndConnectivity The boundary connectivity flag.
//!
void marchingCubes(const ConstArrayAccessor3<double>& grid,
                   const Vector3D& gridSize, const Vector3D& origin,
                   TriangleMesh3* mesh, double isoValue = 0,
                   int bndClose = kDirectionAll,
                   int bndConnectivity = kDirectionNone);

}  // namespace jet

#endif  // INCLUDE_JET_MARCHING_CUBES_H_
