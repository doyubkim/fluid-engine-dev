// Copyright (c) 2018 Doyub Kim
//
// I am making my contributions/submissions to this project solely in my
// personal capacity and am not conveying any rights to any intellectual
// property of any third parties.

#include <pch.h>
#include <jet/advection_solver2.h>
#include <limits>

using namespace jet;

AdvectionSolver2::AdvectionSolver2() {
}

AdvectionSolver2::~AdvectionSolver2() {
}

void AdvectionSolver2::advect(
    const CollocatedVectorGrid2& source,
    const VectorField2& flow,
    double dt,
    CollocatedVectorGrid2* target,
    const ScalarField2& boundarySdf) {
    UNUSED_VARIABLE(source);
    UNUSED_VARIABLE(flow);
    UNUSED_VARIABLE(dt);
    UNUSED_VARIABLE(target);
    UNUSED_VARIABLE(boundarySdf);
}

void AdvectionSolver2::advect(
    const FaceCenteredGrid2& source,
    const VectorField2& flow,
    double dt,
    FaceCenteredGrid2* target,
    const ScalarField2& boundarySdf) {
    UNUSED_VARIABLE(source);
    UNUSED_VARIABLE(flow);
    UNUSED_VARIABLE(dt);
    UNUSED_VARIABLE(target);
    UNUSED_VARIABLE(boundarySdf);
}
