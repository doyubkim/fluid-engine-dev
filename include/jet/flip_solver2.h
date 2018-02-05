// Copyright (c) 2018 Doyub Kim
//
// I am making my contributions/submissions to this project solely in my
// personal capacity and am not conveying any rights to any intellectual
// property of any third parties.

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

    //! Returns the PIC blending factor.
    double picBlendingFactor() const;

    //!
    //! \brief  Sets the PIC blending factor.
    //!
    //! This function sets the PIC blendinf factor which mixes FLIP and PIC
    //! results when transferring velocity from grids to particles in order to
    //! reduce the noise. The factor can be a value between 0 and 1, where 0
    //! means no blending and 1 means full PIC. Default is 0.
    //!
    //! \param[in]  factor The blending factor.
    //!
    void setPicBlendingFactor(double factor);

    //! Returns builder fox FlipSolver2.
    static Builder builder();

 protected:
    //! Transfers velocity field from particles to grids.
    void transferFromParticlesToGrids() override;

    //! Transfers velocity field from grids to particles.
    void transferFromGridsToParticles() override;

 private:
    double _picBlendingFactor = 0.0;
    Array2<float> _uDelta;
    Array2<float> _vDelta;
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
