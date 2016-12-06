// Copyright (c) 2016 Doyub Kim

#ifndef INCLUDE_JET_MARCHING_CUBES_H_
#define INCLUDE_JET_MARCHING_CUBES_H_

#include <jet/array_accessor3.h>
#include <jet/constants.h>
#include <jet/triangle_mesh3.h>

namespace jet {

//!
//! \brief      Computes marching cubes and extract triangle mesh from grid.
//!
//! This function comptues the marching cube algorithm to extract triangle mesh
//! from the scalar grid field. The triangle mesh will be the iso surface, and
//! the iso value can be specified. For the boundaries (or the walls), it can be
//! specified wheather to close or open.
//!
//! \param[in]  grid     The grid.
//! \param[in]  gridSize The grid size.
//! \param[in]  origin   The origin.
//! \param      mesh     The output triangle mesh.
//! \param[in]  isoValue The iso-surface value.
//! \param[in]  bndFlag  The boundary direction flag.
//!
void marchingCubes(
    const ConstArrayAccessor3<double>& grid,
    const Vector3D& gridSize,
    const Vector3D& origin,
    TriangleMesh3* mesh,
    double isoValue = 0,
    int bndFlag = kDirectionAll);

}  // namespace jet

#endif  // INCLUDE_JET_MARCHING_CUBES_H_
