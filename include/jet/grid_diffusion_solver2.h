// Copyright (c) 2016 Doyub Kim

#ifndef INCLUDE_JET_GRID_DIFFUSION_SOLVER2_H_
#define INCLUDE_JET_GRID_DIFFUSION_SOLVER2_H_

#include <jet/collocated_vector_grid2.h>
#include <jet/constant_scalar_field2.h>
#include <jet/constants.h>
#include <jet/face_centered_grid2.h>
#include <jet/scalar_grid2.h>
#include <limits>
#include <memory>

namespace jet {

class GridDiffusionSolver2 {
 public:
    GridDiffusionSolver2();

    virtual ~GridDiffusionSolver2();

    virtual void solve(
        const ScalarGrid2& source,
        double diffusionCoefficient,
        double timeIntervalInSeconds,
        ScalarGrid2* dest,
        const ScalarField2& boundarySdf = ConstantScalarField2(kMaxD),
        const ScalarField2& fluidSdf = ConstantScalarField2(-kMaxD)) = 0;

    virtual void solve(
        const CollocatedVectorGrid2& source,
        double diffusionCoefficient,
        double timeIntervalInSeconds,
        CollocatedVectorGrid2* dest,
        const ScalarField2& boundarySdf = ConstantScalarField2(kMaxD),
        const ScalarField2& fluidSdf = ConstantScalarField2(-kMaxD)) = 0;

    virtual void solve(
        const FaceCenteredGrid2& source,
        double diffusionCoefficient,
        double timeIntervalInSeconds,
        FaceCenteredGrid2* dest,
        const ScalarField2& boundarySdf = ConstantScalarField2(kMaxD),
        const ScalarField2& fluidSdf = ConstantScalarField2(-kMaxD)) = 0;
};

typedef std::shared_ptr<GridDiffusionSolver2> GridDiffusionSolver2Ptr;

}  // namespace jet

#endif  // INCLUDE_JET_GRID_DIFFUSION_SOLVER2_H_
