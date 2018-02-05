// Copyright (c) 2018 Doyub Kim
//
// I am making my contributions/submissions to this project solely in my
// personal capacity and am not conveying any rights to any intellectual
// property of any third parties.

#ifndef INCLUDE_JET_GRID_SINGLE_PHASE_PRESSURE_SOLVER3_H_
#define INCLUDE_JET_GRID_SINGLE_PHASE_PRESSURE_SOLVER3_H_

#include <jet/fdm_linear_system_solver3.h>
#include <jet/fdm_mg_linear_system3.h>
#include <jet/fdm_mg_solver3.h>
#include <jet/grid_boundary_condition_solver3.h>
#include <jet/grid_pressure_solver3.h>

#include <memory>

namespace jet {

//!
//! \brief 3-D single-phase pressure solver.
//!
//! This class implements 3-D single-phase pressure solver. This solver encodes
//! the boundaries like Lego blocks -- if a grid cell center is inside or
//! outside the boundaries, it is either marked as occupied or not.
//! In addition, this class solves single-phase flow, solving the pressure for
//! selected fluid region only and treat other area as an atmosphere region.
//! Thus, the pressure outside the fluid will be set to a constant value and
//! velocity field won't be altered. This solver also computes the fluid
//! boundary in block-like manner; If a grid cell is inside or outside the
//! fluid, it is marked as either fluid or atmosphere. Thus, this solver in
//! general, does not compute subgrid structure.
//!
class GridSinglePhasePressureSolver3 : public GridPressureSolver3 {
 public:
    //! Default constructor.
    GridSinglePhasePressureSolver3();

    //! Default destructor.
    virtual ~GridSinglePhasePressureSolver3();

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
    //! \param[in]    useCompressed         True if it uses compressed system.
    //!
    void solve(
        const FaceCenteredGrid3& input, double timeIntervalInSeconds,
        FaceCenteredGrid3* output,
        const ScalarField3& boundarySdf = ConstantScalarField3(kMaxD),
        const VectorField3& boundaryVelocity = ConstantVectorField3({0, 0, 0}),
        const ScalarField3& fluidSdf = ConstantScalarField3(-kMaxD),
        bool useCompressed = false) override;

    //!
    //! \brief Returns the best boundary condition solver for this solver.
    //!
    //! This function returns the best boundary condition solver that works well
    //! with this pressure solver. Depending on the pressure solver
    //! implementation, different boundary condition solver might be used. For
    //! this particular class, an instance of
    //! GridBlockedBoundaryConditionSolver2 will be returned since this pressure
    //! solver encodes boundaries like pixelated Lego blocks.
    //!
    GridBoundaryConditionSolver3Ptr suggestedBoundaryConditionSolver()
        const override;

    //! Returns the linear system solver.
    const FdmLinearSystemSolver3Ptr& linearSystemSolver() const;

    //! Sets the linear system solver.
    void setLinearSystemSolver(const FdmLinearSystemSolver3Ptr& solver);

    //! Returns the pressure field.
    const FdmVector3& pressure() const;

 private:
    FdmLinearSystem3 _system;
    FdmCompressedLinearSystem3 _compSystem;
    FdmLinearSystemSolver3Ptr _systemSolver;

    FdmMgLinearSystem3 _mgSystem;
    FdmMgSolver3Ptr _mgSystemSolver;

    std::vector<Array3<char>> _markers;

    void buildMarkers(
        const Size3& size,
        const std::function<Vector3D(size_t, size_t, size_t)>& pos,
        const ScalarField3& boundarySdf, const ScalarField3& fluidSdf);

    void decompressSolution();

    virtual void buildSystem(const FaceCenteredGrid3& input,
                             bool useCompressed);

    virtual void applyPressureGradient(const FaceCenteredGrid3& input,
                                       FaceCenteredGrid3* output);
};

//! Shared pointer type for the GridSinglePhasePressureSolver3.
typedef std::shared_ptr<GridSinglePhasePressureSolver3>
    GridSinglePhasePressureSolver3Ptr;

}  // namespace jet

#endif  // INCLUDE_JET_GRID_SINGLE_PHASE_PRESSURE_SOLVER3_H_
