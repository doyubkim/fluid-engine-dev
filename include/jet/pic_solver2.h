// Copyright (c) 2018 Doyub Kim
//
// I am making my contributions/submissions to this project solely in my
// personal capacity and am not conveying any rights to any intellectual
// property of any third parties.

#ifndef INCLUDE_JET_PIC_SOLVER2_H_
#define INCLUDE_JET_PIC_SOLVER2_H_

#include <jet/grid_fluid_solver2.h>
#include <jet/particle_emitter2.h>
#include <jet/particle_system_data2.h>

namespace jet {

//!
//! \brief 2-D Particle-in-Cell (PIC) implementation.
//!
//! This class implements 2-D Particle-in-Cell (PIC) method by inheriting
//! GridFluidSolver2. Since it is a grid-particle hybrid method, the solver
//! also has a particle system to track fluid particles.
//!
//! \see Zhu, Yongning, and Robert Bridson. "Animating sand as a fluid."
//!     ACM Transactions on Graphics (TOG). Vol. 24. No. 3. ACM, 2005.
//!
class PicSolver2 : public GridFluidSolver2 {
 public:
    class Builder;

    //! Default constructor.
    PicSolver2();

    //! Constructs solver with initial grid size.
    PicSolver2(
        const Size2& resolution,
        const Vector2D& gridSpacing,
        const Vector2D& gridOrigin);

    //! Default destructor.
    virtual ~PicSolver2();

    //! Returns the signed-distance field of particles.
    ScalarGrid2Ptr signedDistanceField() const;

    //! Returns the particle system data.
    const ParticleSystemData2Ptr& particleSystemData() const;

    //! Returns the particle emitter.
    const ParticleEmitter2Ptr& particleEmitter() const;

    //! Sets the particle emitter.
    void setParticleEmitter(const ParticleEmitter2Ptr& newEmitter);

    //! Returns builder fox PicSolver2.
    static Builder builder();

 protected:
    Array2<char> _uMarkers;
    Array2<char> _vMarkers;

    //! Initializes the simulator.
    void onInitialize() override;

    //! Invoked before a simulation time-step begins.
    void onBeginAdvanceTimeStep(double timeIntervalInSeconds) override;

    //! Computes the advection term of the fluid solver.
    void computeAdvection(double timeIntervalInSeconds) override;

    //! Returns the signed-distance field of the fluid.
    ScalarField2Ptr fluidSdf() const override;

    //! Transfers velocity field from particles to grids.
    virtual void transferFromParticlesToGrids();

    //! Transfers velocity field from grids to particles.
    virtual void transferFromGridsToParticles();

    //! Moves particles.
    virtual void moveParticles(double timeIntervalInSeconds);

 private:
    size_t _signedDistanceFieldId;
    ParticleSystemData2Ptr _particles;
    ParticleEmitter2Ptr _particleEmitter;

    void extrapolateVelocityToAir();

    void buildSignedDistanceField();

    void updateParticleEmitter(double timeIntervalInSeconds);
};

//! Shared pointer type for the PicSolver2.
typedef std::shared_ptr<PicSolver2> PicSolver2Ptr;


//!
//! \brief Front-end to create PicSolver2 objects step by step.
//!
class PicSolver2::Builder final
    : public GridFluidSolverBuilderBase2<PicSolver2::Builder> {
 public:
    //! Builds PicSolver2.
    PicSolver2 build() const;

    //! Builds shared pointer of PicSolver2 instance.
    PicSolver2Ptr makeShared() const;
};

}  // namespace jet

#endif  // INCLUDE_JET_PIC_SOLVER2_H_
