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
    //! Default constructor.
    FlipSolver3();

    //! Default destructor.
    virtual ~FlipSolver3();

 protected:
    //! Transfers velocity field from particles to grids.
    void transferFromParticlesToGrids() override;

    //! Transfers velocity field from grids to particles.
    void transferFromGridsToParticles() override;

 private:
    FaceCenteredGrid3 _delta;
};

}  // namespace jet

#endif  // INCLUDE_JET_FLIP_SOLVER3_H_
