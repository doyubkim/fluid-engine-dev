// Copyright (c) 2016 Doyub Kim

#ifndef INCLUDE_JET_GRID_PRESSURE_SOLVER3_H_
#define INCLUDE_JET_GRID_PRESSURE_SOLVER3_H_

#include <jet/collocated_vector_grid3.h>
#include <jet/constant_scalar_field3.h>
#include <jet/constant_vector_field3.h>
#include <jet/constants.h>
#include <jet/face_centered_grid3.h>
#include <jet/grid_boundary_condition_solver3.h>
#include <jet/scalar_grid3.h>
#include <memory>

namespace jet {

//!
//! \brief Abstract base class for 2-D grid-based pressure solver.
//!
//! This class represents a 2-D grid-based pressure solver interface which can
//! be used as a sub-step of GridFluidSolver2. Inheriting classes must implement
//! the core GridPressureSolver2::solve function as well as the helper function
//! GridPressureSolver2::suggestedBoundaryConditionSolver.
//!
class GridPressureSolver3 {
 public:
    //! Default constructor.
    GridPressureSolver3();

    //! Default destructor.
    virtual ~GridPressureSolver3();

    //!
    //! \brief Solves the pressure term and apply it to the velocity field.
    //!
    //! This function takes input velocity field and outputs pressure-applied
    //! velocity field. It also accepts extra arguments such as \p boundarySdf
    //! and \p fluidSdf that represent signed-distance representation of the
    //! boundary and fluid area. The negative region of \p boundarySdf means
    //! it is occupied by solid object. Also, the positive / negative area of
    //! the \p fluidSdf means it is occupied by fluid / atmosphere. If not
    //! specified, constant scalar field with kMaxD will be used for
    //! \p boundarySdf meaning that no boundary at all. Similarly, a constant
    //! field with -kMaxD will be used for \p fluidSdf which means it's fully
    //! occupied with fluid without any atmosphere.
    //!
    //! \param[in]    input                 The input velocity field.
    //! \param[in]    timeIntervalInSeconds The time interval for the sim.
    //! \param[inout] output                The output velocity field.
    //! \param[in]    boundarySdf           The SDF of the boundary.
    //! \param[in]    fluidSdf              The SDF of the fluid/atmosphere.
    //!
    virtual void solve(
        const FaceCenteredGrid3& input,
        double timeIntervalInSeconds,
        FaceCenteredGrid3* output,
        const ScalarField3& boundarySdf
            = ConstantScalarField3(kMaxD),
        const VectorField3& boundaryVelocity
            = ConstantVectorField3({0, 0, 0}),
        const ScalarField3& fluidSdf
            = ConstantScalarField3(-kMaxD)) = 0;

    //!
    //! \brief Returns the best boundary condition solver for this solver.
    //!
    //! This function returns the best boundary condition solver that works well
    //! with this pressure solver. Depending on the pressure solver
    //! implementation, different boundary condition solver might be used.
    //!
    virtual GridBoundaryConditionSolver3Ptr
        suggestedBoundaryConditionSolver() const = 0;
};

//! Shared pointer type for the GridPressureSolver3.
typedef std::shared_ptr<GridPressureSolver3> GridPressureSolver3Ptr;

}  // namespace jet

#endif  // INCLUDE_JET_GRID_PRESSURE_SOLVER3_H_
