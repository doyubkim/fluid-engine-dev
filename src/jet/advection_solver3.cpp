// Copyright (c) 2018 Doyub Kim
//
// I am making my contributions/submissions to this project solely in my
// personal capacity and am not conveying any rights to any intellectual
// property of any third parties.

#include <pch.h>
#include <jet/advection_solver3.h>
#include <limits>

using namespace jet;

AdvectionSolver3::AdvectionSolver3() {
}

AdvectionSolver3::~AdvectionSolver3() {
}

void AdvectionSolver3::advect(
    const CollocatedVectorGrid3& source,
    const VectorField3& flow,
    double dt,
    CollocatedVectorGrid3* target,
    const ScalarField3& boundarySdf) {
    UNUSED_VARIABLE(source);
    UNUSED_VARIABLE(flow);
    UNUSED_VARIABLE(dt);
    UNUSED_VARIABLE(target);
    UNUSED_VARIABLE(boundarySdf);
}

void AdvectionSolver3::advect(
    const FaceCenteredGrid3& source,
    const VectorField3& flow,
    double dt,
    FaceCenteredGrid3* target,
    const ScalarField3& boundarySdf) {
    UNUSED_VARIABLE(source);
    UNUSED_VARIABLE(flow);
    UNUSED_VARIABLE(dt);
    UNUSED_VARIABLE(target);
    UNUSED_VARIABLE(boundarySdf);
}
