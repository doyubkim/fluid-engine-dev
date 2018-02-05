// Copyright (c) 2018 Doyub Kim
//
// I am making my contributions/submissions to this project solely in my
// personal capacity and am not conveying any rights to any intellectual
// property of any third parties.

#ifndef INCLUDE_JET_GRID_BACKWARD_EULER_DIFFUSION_SOLVER2_H_
#define INCLUDE_JET_GRID_BACKWARD_EULER_DIFFUSION_SOLVER2_H_

#include <jet/constant_scalar_field2.h>
#include <jet/fdm_linear_system_solver2.h>
#include <jet/grid_diffusion_solver2.h>
#include <limits>
#include <memory>

namespace jet {

//!
//! \brief 2-D grid-based backward Euler diffusion solver.
//!
//! This class implements 2-D grid-based forward Euler diffusion solver using
//! second-order central differencing spatially. Since the method is following
//! the implicit time-integration (i.e. backward Euler), larger time interval or
//! diffusion coefficient can be used without breaking the result. Note, higher
//! values for those parameters will still impact the accuracy of the result.
//! To solve the backward Euler method, a linear system solver is used and
//! incomplete Cholesky conjugate gradient method is used by default.
//!
class GridBackwardEulerDiffusionSolver2 final : public GridDiffusionSolver2 {
 public:
    enum BoundaryType {
        Dirichlet,
        Neumann
    };

    //! Constructs the solver with given boundary type.
    explicit GridBackwardEulerDiffusionSolver2(
        BoundaryType boundaryType = Neumann);

    //!
    //! Solves diffusion equation for a scalar field.
    //!
    //! \param source Input scalar field.
    //! \param diffusionCoefficient Amount of diffusion.
    //! \param timeIntervalInSeconds Small time-interval that diffusion occur.
    //! \param dest Output scalar field.
    //! \param boundarySdf Shape of the solid boundary that is empty by default.
    //! \param boundarySdf Shape of the fluid boundary that is full by default.
    //!
    void solve(
        const ScalarGrid2& source,
        double diffusionCoefficient,
        double timeIntervalInSeconds,
        ScalarGrid2* dest,
        const ScalarField2& boundarySdf
            = ConstantScalarField2(kMaxD),
        const ScalarField2& fluidSdf
            = ConstantScalarField2(-kMaxD)) override;

    //!
    //! Solves diffusion equation for a collocated vector field.
    //!
    //! \param source Input collocated vector field.
    //! \param diffusionCoefficient Amount of diffusion.
    //! \param timeIntervalInSeconds Small time-interval that diffusion occur.
    //! \param dest Output collocated vector field.
    //! \param boundarySdf Shape of the solid boundary that is empty by default.
    //! \param boundarySdf Shape of the fluid boundary that is full by default.
    //!
    void solve(
        const CollocatedVectorGrid2& source,
        double diffusionCoefficient,
        double timeIntervalInSeconds,
        CollocatedVectorGrid2* dest,
        const ScalarField2& boundarySdf
            = ConstantScalarField2(kMaxD),
        const ScalarField2& fluidSdf
            = ConstantScalarField2(-kMaxD)) override;

    //!
    //! Solves diffusion equation for a face-centered vector field.
    //!
    //! \param source Input face-centered vector field.
    //! \param diffusionCoefficient Amount of diffusion.
    //! \param timeIntervalInSeconds Small time-interval that diffusion occur.
    //! \param dest Output face-centered vector field.
    //! \param boundarySdf Shape of the solid boundary that is empty by default.
    //! \param boundarySdf Shape of the fluid boundary that is full by default.
    //!
    void solve(
        const FaceCenteredGrid2& source,
        double diffusionCoefficient,
        double timeIntervalInSeconds,
        FaceCenteredGrid2* dest,
        const ScalarField2& boundarySdf
            = ConstantScalarField2(kMaxD),
        const ScalarField2& fluidSdf
            = ConstantScalarField2(-kMaxD)) override;

    //! Sets the linear system solver for this diffusion solver.
    void setLinearSystemSolver(const FdmLinearSystemSolver2Ptr& solver);

 private:
    BoundaryType _boundaryType;
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

//! Shared pointer type for the GridBackwardEulerDiffusionSolver2.
typedef std::shared_ptr<GridBackwardEulerDiffusionSolver2>
    GridBackwardEulerDiffusionSolver2Ptr;

}  // namespace jet

#endif  // INCLUDE_JET_GRID_BACKWARD_EULER_DIFFUSION_SOLVER2_H_
