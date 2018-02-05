// Copyright (c) 2018 Doyub Kim
//
// I am making my contributions/submissions to this project solely in my
// personal capacity and am not conveying any rights to any intellectual
// property of any third parties.

#ifndef INCLUDE_JET_APIC_SOLVER2_H_
#define INCLUDE_JET_APIC_SOLVER2_H_

#include <jet/pic_solver2.h>

namespace jet {

//!
//! \brief 2-D Affine Particle-in-Cell (APIC) implementation
//!
//! This class implements 2-D Affine Particle-in-Cell (APIC) solver from the
//! SIGGRAPH paper, Jiang 2015.
//!
//! \see Jiang, Chenfanfu, et al. "The affine particle-in-cell method."
//!      ACM Transactions on Graphics (TOG) 34.4 (2015): 51.
//!
class ApicSolver2 : public PicSolver2 {
 public:
    class Builder;

    //! Default constructor.
    ApicSolver2();

    //! Constructs solver with initial grid size.
    ApicSolver2(
        const Size2& resolution,
        const Vector2D& gridSpacing,
        const Vector2D& gridOrigin);

    //! Default destructor.
    virtual ~ApicSolver2();

    //! Returns builder fox ApicSolver2.
    static Builder builder();

 protected:
    //! Transfers velocity field from particles to grids.
    void transferFromParticlesToGrids() override;

    //! Transfers velocity field from grids to particles.
    void transferFromGridsToParticles() override;

 private:
    Array1<Vector2D> _cX;
    Array1<Vector2D> _cY;
};

//! Shared pointer type for the ApicSolver2.
typedef std::shared_ptr<ApicSolver2> ApicSolver2Ptr;


//!
//! \brief Front-end to create ApicSolver2 objects step by step.
//!
class ApicSolver2::Builder final
    : public GridFluidSolverBuilderBase2<ApicSolver2::Builder> {
 public:
    //! Builds ApicSolver2.
    ApicSolver2 build() const;

    //! Builds shared pointer of ApicSolver2 instance.
    ApicSolver2Ptr makeShared() const;
};

}  // namespace jet

#endif  // INCLUDE_JET_APIC_SOLVER2_H_
