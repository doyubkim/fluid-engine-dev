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

//! Abstract base class for 3-D grid-based diffusion equation solver.
class GridDiffusionSolver3 {
 public:
    //! Default constructor.
    GridDiffusionSolver3();

    //! Default destructor.
    virtual ~GridDiffusionSolver3();

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
    virtual void solve(
        const ScalarGrid3& source,
        double diffusionCoefficient,
        double timeIntervalInSeconds,
        ScalarGrid3* dest,
        const ScalarField3& boundarySdf = ConstantScalarField3(kMaxD),
        const ScalarField3& fluidSdf = ConstantScalarField3(-kMaxD)) = 0;

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
    virtual void solve(
        const CollocatedVectorGrid3& source,
        double diffusionCoefficient,
        double timeIntervalInSeconds,
        CollocatedVectorGrid3* dest,
        const ScalarField3& boundarySdf = ConstantScalarField3(kMaxD),
        const ScalarField3& fluidSdf = ConstantScalarField3(-kMaxD)) = 0;

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
