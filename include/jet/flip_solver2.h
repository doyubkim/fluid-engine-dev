// Copyright (c) 2016 Doyub Kim

#ifndef INCLUDE_JET_FLIP_SOLVER2_H_
#define INCLUDE_JET_FLIP_SOLVER2_H_

#include <jet/pic_solver2.h>

namespace jet {

//!
//! \brief 2-D Fluid-Implicit Particle (FLIP) implementation.
//!
//! This class implements 2-D Fluid-Implicit Particle (FLIP) solver from the
//! SIGGRAPH paper, Zhu and Bridson 2005. By transfering delta-velocity field
//! from grid to particles, the FLIP solver achieves less viscous fluid flow
//! compared to the original PIC method.
//!
//! \see Zhu, Yongning, and Robert Bridson. "Animating sand as a fluid."
//!     ACM Transactions on Graphics (TOG). Vol. 24. No. 3. ACM, 2005.
//!
class FlipSolver2 : public PicSolver2 {
 public:
    class Builder;

    //! Default constructor.
    FlipSolver2();

    //! Constructs solver with initial grid size.
    FlipSolver2(
        const Size2& resolution,
        const Vector2D& gridSpacing,
        const Vector2D& gridOrigin);

    //! Default destructor.
    virtual ~FlipSolver2();

    //! Returns builder fox FlipSolver2.
    static Builder builder();

 protected:
    //! Transfers velocity field from particles to grids.
    void transferFromParticlesToGrids() override;

    //! Transfers velocity field from grids to particles.
    void transferFromGridsToParticles() override;

 private:
    FaceCenteredGrid2 _delta;
};

//! Shared pointer type for the FlipSolver2.
typedef std::shared_ptr<FlipSolver2> FlipSolver2Ptr;


//!
//! \brief Front-end to create FlipSolver2 objects step by step.
//!
class FlipSolver2::Builder final
    : public GridFluidSolverBuilderBase2<FlipSolver2::Builder> {
 public:
    //! Builds FlipSolver2.
    FlipSolver2 build() const;

    //! Builds shared pointer of FlipSolver2 instance.
    FlipSolver2Ptr makeShared() const;
};

}  // namespace jet

#endif  // INCLUDE_JET_FLIP_SOLVER2_H_
