// Copyright (c) 2016 Doyub Kim

#ifndef INCLUDE_JET_GRID_BACKWARD_EULER_DIFFUSION_SOLVER2_H_
#define INCLUDE_JET_GRID_BACKWARD_EULER_DIFFUSION_SOLVER2_H_

#include <jet/constant_scalar_field2.h>
#include <jet/fdm_linear_system_solver2.h>
#include <jet/grid_diffusion_solver2.h>
#include <limits>
#include <memory>

namespace jet {

class GridBackwardEulerDiffusionSolver2 final : public GridDiffusionSolver2 {
 public:
    enum BoundaryType {
        Dirichlet,
        Neumann
    };

    explicit GridBackwardEulerDiffusionSolver2(
        BoundaryType boundaryType = Neumann);

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

    void setLinearSystemSolver(const FdmLinearSystemSolver2Ptr& solver);

 private:
    BoundaryType _boundaryType = Dirichlet;
    FdmLinearSystem2 _system;
    FdmLinearSystemSolver2Ptr _systemSolver;
    Array2<char> _markers;

    void buildMarkers(
        const Size2& size,
        const std::function<Vector2D(size_t, size_t)>& pos,
        const ScalarField2& boundarySdf,
        const ScalarField2& fluidSdf);

    void buildMatrix(
        const Size2& size,
        const Vector2D& c);

    void buildVectors(
        const ConstArrayAccessor2<double>& f,
        const Vector2D& c);

    void buildVectors(
        const ConstArrayAccessor2<Vector2D>& f,
        const Vector2D& c,
        size_t component);
};

typedef std::shared_ptr<GridBackwardEulerDiffusionSolver2>
    GridBackwardEulerDiffusionSolver2Ptr;

}  // namespace jet

#endif  // INCLUDE_JET_GRID_BACKWARD_EULER_DIFFUSION_SOLVER2_H_
