// Copyright (c) 2016 Doyub Kim

#ifndef INCLUDE_JET_GRID_FORWARD_EULER_DIFFUSION_SOLVER2_H_
#define INCLUDE_JET_GRID_FORWARD_EULER_DIFFUSION_SOLVER2_H_

#include <jet/constant_scalar_field2.h>
#include <jet/grid_diffusion_solver2.h>
#include <limits>
#include <memory>

namespace jet {

class GridForwardEulerDiffusionSolver2 final : public GridDiffusionSolver2 {
 public:
    GridForwardEulerDiffusionSolver2();

    void solve(
        const ScalarGrid2& source,
        double diffusionCoefficient,
        double timeIntervalInSeconds,
        ScalarGrid2* dest,
        const ScalarField2& boundarySdf
            = ConstantScalarField2(kMaxD),
        const ScalarField2& fluidSdf
            = ConstantScalarField2(-kMaxD)) override;

    void solve(
        const CollocatedVectorGrid2& source,
        double diffusionCoefficient,
        double timeIntervalInSeconds,
        CollocatedVectorGrid2* dest,
        const ScalarField2& boundarySdf
            = ConstantScalarField2(kMaxD),
        const ScalarField2& fluidSdf
            = ConstantScalarField2(-kMaxD)) override;

    void solve(
        const FaceCenteredGrid2& source,
        double diffusionCoefficient,
        double timeIntervalInSeconds,
        FaceCenteredGrid2* dest,
        const ScalarField2& boundarySdf
            = ConstantScalarField2(kMaxD),
        const ScalarField2& fluidSdf
            = ConstantScalarField2(-kMaxD)) override;

 private:
    Array2<char> _markers;

    void buildMarkers(
        const Size2& size,
        const std::function<Vector2D(size_t, size_t)>& pos,
        const ScalarField2& boundarySdf,
        const ScalarField2& fluidSdf);
};

typedef std::shared_ptr<GridForwardEulerDiffusionSolver2>
    GridForwardEulerDiffusionSolver2Ptr;

}  // namespace jet

#endif  // INCLUDE_JET_GRID_FORWARD_EULER_DIFFUSION_SOLVER2_H_
