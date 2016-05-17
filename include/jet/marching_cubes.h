// Copyright (c) 2016 Doyub Kim

#ifndef INCLUDE_JET_MARCHING_CUBES_H_
#define INCLUDE_JET_MARCHING_CUBES_H_

#include <jet/array_accessor3.h>
#include <jet/triangle_mesh3.h>

namespace jet {

static const int kMarchingCubesBoundaryFlagNone = 0;
static const int kMarchingCubesBoundaryFlagLeft = 1 << 0;
static const int kMarchingCubesBoundaryFlagRight = 1 << 1;
static const int kMarchingCubesBoundaryFlagDown = 1 << 2;
static const int kMarchingCubesBoundaryFlagUp = 1 << 3;
static const int kMarchingCubesBoundaryFlagBack = 1 << 4;
static const int kMarchingCubesBoundaryFlagFront = 1 << 5;
static const int kMarchingCubesBoundaryFlagAll
    = kMarchingCubesBoundaryFlagLeft
    | kMarchingCubesBoundaryFlagRight
    | kMarchingCubesBoundaryFlagDown
    | kMarchingCubesBoundaryFlagUp
    | kMarchingCubesBoundaryFlagBack
    | kMarchingCubesBoundaryFlagFront;

void marchingCubes(
    const ConstArrayAccessor3<double>& grid,
    const Vector3D& gridSize,
    const Vector3D& origin,
    TriangleMesh3* mesh,
    double isoValue = 0,
    int bndFlag = kMarchingCubesBoundaryFlagAll);

}  // namespace jet

#endif  // INCLUDE_JET_MARCHING_CUBES_H_
