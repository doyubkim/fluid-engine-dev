// Copyright (c) 2016 Doyub Kim

#ifndef INCLUDE_JET_FLIP_SOLVER3_H_
#define INCLUDE_JET_FLIP_SOLVER3_H_

#include <jet/pic_solver3.h>

namespace jet {

//!
//! \brief 3-D Fluid-Implicit Particle (FLIP) implementation.
//!
//! This class implements 3-D Fluid-Implicit Particle (FLIP) solver from the
//! SIGGRAPH paper, Zhu and Bridson 2005. By transfering delta-velocity field
//! from grid to particles, the FLIP solver achieves less viscous fluid flow
//! compared to the original PIC method.
//!
//! \see Zhu, Yongning, and Robert Bridson. "Animating sand as a fluid."
//!     ACM Transactions on Graphics (TOG). Vol. 24. No. 3. ACM, 2005.
//!
class FlipSolver3 : public PicSolver3 {
 public:
    class Builder;

    //! Default constructor.
    FlipSolver3();

    //! Constructs solver with initial grid size.
    FlipSolver3(
        const Size3& resolution,
        const Vector3D& gridSpacing,
        const Vector3D& gridOrigin);

    //! Default destructor.
    virtual ~FlipSolver3();

    //! Returns builder fox FlipSolver3.
    static Builder builder();

 protected:
    //! Transfers velocity field from particles to grids.
    void transferFromParticlesToGrids() override;

    //! Transfers velocity field from grids to particles.
    void transferFromGridsToParticles() override;

 private:
    FaceCenteredGrid3 _delta;
};

//! Shared pointer type for the FlipSolver3.
typedef std::shared_ptr<FlipSolver3> FlipSolver3Ptr;


//!
//! \brief Front-end to create FlipSolver3 objects step by step.
//!
class FlipSolver3::Builder final
    : public GridFluidSolverBuilderBase3<FlipSolver3::Builder> {
 public:
    //! Builds FlipSolver3.
    FlipSolver3 build() const;

    //! Builds shared pointer of FlipSolver3 instance.
    FlipSolver3Ptr makeShared() const;
};

}  // namespace jet

#endif  // INCLUDE_JET_FLIP_SOLVER3_H_
