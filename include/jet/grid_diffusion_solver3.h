// Copyright (c) 2016 Doyub Kim

#ifndef INCLUDE_JET_GRID_DIFFUSION_SOLVER3_H_
#define INCLUDE_JET_GRID_DIFFUSION_SOLVER3_H_

#include <jet/collocated_vector_grid3.h>
#include <jet/constant_scalar_field3.h>
#include <jet/constants.h>
#include <jet/face_centered_grid3.h>
#include <jet/scalar_grid3.h>
#include <limits>
#include <memory>

namespace jet {

class GridDiffusionSolver3 {
 public:
    GridDiffusionSolver3();

    virtual ~GridDiffusionSolver3();

    virtual void solve(
        const ScalarGrid3& source,
        double diffusionCoefficient,
        double timeIntervalInSeconds,
        ScalarGrid3* dest,
        const ScalarField3& boundarySdf = ConstantScalarField3(kMaxD),
        const ScalarField3& fluidSdf = ConstantScalarField3(-kMaxD)) = 0;

    virtual void solve(
        const CollocatedVectorGrid3& source,
        double diffusionCoefficient,
        double timeIntervalInSeconds,
        CollocatedVectorGrid3* dest,
        const ScalarField3& boundarySdf = ConstantScalarField3(kMaxD),
        const ScalarField3& fluidSdf = ConstantScalarField3(-kMaxD)) = 0;

    virtual void solve(
        const FaceCenteredGrid3& source,
        double diffusionCoefficient,
        double timeIntervalInSeconds,
        FaceCenteredGrid3* dest,
        const ScalarField3& boundarySdf = ConstantScalarField3(kMaxD),
        const ScalarField3& fluidSdf = ConstantScalarField3(-kMaxD)) = 0;
};

typedef std::shared_ptr<GridDiffusionSolver3> GridDiffusionSolver3Ptr;

}  // namespace jet

#endif  // INCLUDE_JET_GRID_DIFFUSION_SOLVER3_H_
