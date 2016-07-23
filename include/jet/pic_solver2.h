// Copyright (c) 2016 Doyub Kim

#ifndef INCLUDE_JET_PIC_SOLVER2_H_
#define INCLUDE_JET_PIC_SOLVER2_H_

#include <jet/grid_fluid_solver2.h>
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
    //! Default constructor.
    PicSolver2();

    //! Default destructor.
    virtual ~PicSolver2();

    //! Returns the signed-distance field of particles.
    ScalarGrid2Ptr signedDistanceField() const;

    //! Returns the particle system data.
    const ParticleSystemData2Ptr& particleSystemData() const;

 protected:
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

    Array2<char> _uMarkers;
    Array2<char> _vMarkers;

    void extrapolateVelocityToAir();

    void buildSignedDistanceField();
};

}  // namespace jet

#endif  // INCLUDE_JET_PIC_SOLVER2_H_
