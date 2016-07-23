// Copyright (c) 2016 Doyub Kim

#ifndef INCLUDE_JET_GRID_FORWARD_EULER_DIFFUSION_SOLVER3_H_
#define INCLUDE_JET_GRID_FORWARD_EULER_DIFFUSION_SOLVER3_H_

#include <jet/constant_scalar_field3.h>
#include <jet/grid_diffusion_solver3.h>
#include <limits>
#include <memory>

namespace jet {

//!
//! \brief 3-D grid-based forward Euler diffusion solver.
//!
//! This class implements 3-D grid-based forward Euler diffusion solver using
//! second-order central differencing spatially. Since the method is relying on
//! explicit time-integration (i.e. foward Euler), the diffusion coefficient is
//! limited by the time interval and grid spacing such as:
//! \f$\mu < \frac{h}{12\Delta t} \f$ where \f$\mu\f$, \f$h\f$, and
//! \f$\Delta t\f$ are the diffusion coefficient, grid spacing, and time
//! interval, respectively.
//!
class GridForwardEulerDiffusionSolver3 final : public GridDiffusionSolver3 {
 public:
    //! Default constructor.
    GridForwardEulerDiffusionSolver3();

    //!
    //! \brief Solves diffusion equation for a scalar field.
    //!
    //! This function solves diffusion equation for given scalar field \p source
    //! and store the result to \p dest. The target equation can be written as
    //! \f$\frac{\partial f}{\partial t} = \mu\nabla^2 f\f$ where \f$\mu\f$ is
    //! the diffusion coefficient.
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
    //! \brief Solves diffusion equation for a collocated vector field.
    //!
    //! This function solves diffusion equation for given collocated vector
    //! field \p source and store the result to \p dest. The target equation can
    //! be written as \f$\frac{\partial f}{\partial t} = \mu\nabla^2 f\f$ where
    //! \f$\mu\f$ is the diffusion coefficient.
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
    //! \brief Solves diffusion equation for a face-centered vector field.
    //!
    //! This function solves diffusion equation for given face-centered vector
    //! field \p source and store the result to \p dest. The target equation can
    //! be written as \f$\frac{\partial f}{\partial t} = \mu\nabla^2 f\f$ where
    //! \f$\mu\f$ is the diffusion coefficient.
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

 private:
    Array3<char> _markers;

    void buildMarkers(
        const Size3& size,
        const std::function<Vector3D(size_t, size_t, size_t)>& pos,
        const ScalarField3& boundarySdf,
        const ScalarField3& fluidSdf);
};

typedef std::shared_ptr<GridForwardEulerDiffusionSolver3>
    GridForwardEulerDiffusionSolver3Ptr;

}  // namespace jet

#endif  // INCLUDE_JET_GRID_FORWARD_EULER_DIFFUSION_SOLVER3_H_
