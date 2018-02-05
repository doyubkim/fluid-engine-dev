// Copyright (c) 2018 Doyub Kim
//
// I am making my contributions/submissions to this project solely in my
// personal capacity and am not conveying any rights to any intellectual
// property of any third parties.

#ifndef INCLUDE_JET_GRID_BACKWARD_EULER_DIFFUSION_SOLVER3_H_
#define INCLUDE_JET_GRID_BACKWARD_EULER_DIFFUSION_SOLVER3_H_

#include <jet/constant_scalar_field3.h>
#include <jet/fdm_linear_system_solver3.h>
#include <jet/grid_diffusion_solver3.h>
#include <limits>
#include <memory>

namespace jet {

//!
//! \brief 3-D grid-based backward Euler diffusion solver.
//!
//! This class implements 3-D grid-based forward Euler diffusion solver using
//! second-order central differencing spatially. Since the method is following
//! the implicit time-integration (i.e. backward Euler), larger time interval or
//! diffusion coefficient can be used without breaking the result. Note, higher
//! values for those parameters will still impact the accuracy of the result.
//! To solve the backward Euler method, a linear system solver is used and
//! incomplete Cholesky conjugate gradient method is used by default.
//!
class GridBackwardEulerDiffusionSolver3 final : public GridDiffusionSolver3 {
 public:
    enum BoundaryType {
        Dirichlet,
        Neumann
    };

    //! Constructs the solver with given boundary type.
    explicit GridBackwardEulerDiffusionSolver3(
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
        const ScalarGrid3& source,
        double diffusionCoefficient,
        double timeIntervalInSeconds,
        ScalarGrid3* dest,
        const ScalarField3& boundarySdf
            = ConstantScalarField3(kMaxD),
        const ScalarField3& fluidSdf
            = ConstantScalarField3(-kMaxD)) override;

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
        const CollocatedVectorGrid3& source,
        double diffusionCoefficient,
        double timeIntervalInSeconds,
        CollocatedVectorGrid3* dest,
        const ScalarField3& boundarySdf
            = ConstantScalarField3(kMaxD),
        const ScalarField3& fluidSdf
            = ConstantScalarField3(-kMaxD)) override;

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
        const FaceCenteredGrid3& source,
        double diffusionCoefficient,
        double timeIntervalInSeconds,
        FaceCenteredGrid3* dest,
        const ScalarField3& boundarySdf
            = ConstantScalarField3(kMaxD),
        const ScalarField3& fluidSdf
            = ConstantScalarField3(-kMaxD)) override;

    //! Sets the linear system solver for this diffusion solver.
    void setLinearSystemSolver(const FdmLinearSystemSolver3Ptr& solver);

 private:
    BoundaryType _boundaryType;
    FdmLinearSystem3 _system;
    FdmLinearSystemSolver3Ptr _systemSolver;
    Array3<char> _markers;

    void buildMarkers(
        const Size3& size,
        const std::function<Vector3D(size_t, size_t, size_t)>& pos,
        const ScalarField3& boundarySdf,
        const ScalarField3& fluidSdf);

    void buildMatrix(
        const Size3& size,
        const Vector3D& c);

    void buildVectors(
        const ConstArrayAccessor3<double>& f,
        const Vector3D& c);

    void buildVectors(
        const ConstArrayAccessor3<Vector3D>& f,
        const Vector3D& c,
        size_t component);
};

//! Shared pointer type for the GridBackwardEulerDiffusionSolver3.
typedef std::shared_ptr<GridBackwardEulerDiffusionSolver3>
    GridBackwardEulerDiffusionSolver3Ptr;

}  // namespace jet

#endif  // INCLUDE_JET_GRID_BACKWARD_EULER_DIFFUSION_SOLVER3_H_
