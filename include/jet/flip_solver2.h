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
    //! Default constructor.
    FlipSolver2();

    //! Default destructor.
    virtual ~FlipSolver2();

 protected:
    //! Transfers velocity field from particles to grids.
    void transferFromParticlesToGrids() override;

    //! Transfers velocity field from grids to particles.
    void transferFromGridsToParticles() override;

 private:
    FaceCenteredGrid2 _delta;
};

}  // namespace jet

#endif  // INCLUDE_JET_FLIP_SOLVER2_H_
