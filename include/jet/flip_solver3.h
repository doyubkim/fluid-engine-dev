// Copyright (c) 2018 Doyub Kim
//
// I am making my contributions/submissions to this project solely in my
// personal capacity and am not conveying any rights to any intellectual
// property of any third parties.

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

    //! Returns builder fox FlipSolver3.
    static Builder builder();

 protected:
    //! Transfers velocity field from particles to grids.
    void transferFromParticlesToGrids() override;

    //! Transfers velocity field from grids to particles.
    void transferFromGridsToParticles() override;

 private:
    double _picBlendingFactor = 0.0;
    Array3<float> _uDelta;
    Array3<float> _vDelta;
    Array3<float> _wDelta;
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
