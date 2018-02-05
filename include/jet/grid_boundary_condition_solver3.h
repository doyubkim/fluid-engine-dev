// Copyright (c) 2018 Doyub Kim
//
// I am making my contributions/submissions to this project solely in my
// personal capacity and am not conveying any rights to any intellectual
// property of any third parties.

#ifndef INCLUDE_JET_GRID_BOUNDARY_CONDITION_SOLVER3_H_
#define INCLUDE_JET_GRID_BOUNDARY_CONDITION_SOLVER3_H_

#include <jet/collider3.h>
#include <jet/constants.h>
#include <jet/face_centered_grid3.h>
#include <jet/scalar_field3.h>

#include <memory>

namespace jet {

//!
//! \brief Abstract base class for 3-D boundary condition solver for grids.
//!
//! This is a helper class to constrain the 3-D velocity field with given
//! collider object. It also determines whether to open any domain boundaries.
//! To control the friction level, tune the collider parameter.
//!
class GridBoundaryConditionSolver3 {
 public:
    //! Default constructor.
    GridBoundaryConditionSolver3();

    //! Default destructor.
    virtual ~GridBoundaryConditionSolver3();

    //! Returns associated collider.
    const Collider3Ptr& collider() const;

    //!
    //! \brief Applies new collider and build the internals.
    //!
    //! This function is called to apply new collider and build the internal
    //! cache. To provide a hint to the cache, info for the expected velocity
    //! grid that will be constrained is provided.
    //!
    //! \param newCollider New collider to apply.
    //! \param gridSize Size of the velocity grid to be constrained.
    //! \param gridSpacing Grid spacing of the velocity grid to be constrained.
    //! \param gridOrigin Origin of the velocity grid to be constrained.
    //!
    void updateCollider(
        const Collider3Ptr& newCollider,
        const Size3& gridSize,
        const Vector3D& gridSpacing,
        const Vector3D& gridOrigin);

    //! Returns the closed domain boundary flag.
    int closedDomainBoundaryFlag() const;

    //! Sets the closed domain boundary flag.
    void setClosedDomainBoundaryFlag(int flag);

    //!
    //! Constrains the velocity field to conform the collider boundary.
    //!
    //! \param velocity Input and output velocity grid.
    //! \param extrapolationDepth Number of inner-collider grid cells that
    //!     velocity will get extrapolated.
    //!
    virtual void constrainVelocity(
        FaceCenteredGrid3* velocity,
        unsigned int extrapolationDepth = 5) = 0;

    //! Returns the signed distance field of the collider.
    virtual ScalarField3Ptr colliderSdf() const = 0;

    //! Returns the velocity field of the collider.
    virtual VectorField3Ptr colliderVelocityField() const = 0;

 protected:
    //! Invoked when a new collider is set.
    virtual void onColliderUpdated(
        const Size3& gridSize,
        const Vector3D& gridSpacing,
        const Vector3D& gridOrigin) = 0;

    //! Returns the size of the velocity grid to be constrained.
    const Size3& gridSize() const;

    //! Returns the spacing of the velocity grid to be constrained.
    const Vector3D& gridSpacing() const;

    //! Returns the origin of the velocity grid to be constrained.
    const Vector3D& gridOrigin() const;

 private:
    Collider3Ptr _collider;
    Size3 _gridSize;
    Vector3D _gridSpacing;
    Vector3D _gridOrigin;
    int _closedDomainBoundaryFlag = kDirectionAll;
};

//! Shared pointer type for the GridBoundaryConditionSolver3.
typedef std::shared_ptr<GridBoundaryConditionSolver3>
    GridBoundaryConditionSolver3Ptr;

}  // namespace jet

#endif  // INCLUDE_JET_GRID_BOUNDARY_CONDITION_SOLVER3_H_
