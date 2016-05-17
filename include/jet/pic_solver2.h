// Copyright (c) 2016 Doyub Kim

#ifndef INCLUDE_JET_PIC_SOLVER2_H_
#define INCLUDE_JET_PIC_SOLVER2_H_

#include <jet/grid_fluid_solver2.h>
#include <jet/particle_system_data2.h>

namespace jet {

class PicSolver2 : public GridFluidSolver2 {
 public:
    PicSolver2();

    virtual ~PicSolver2();

    ScalarGrid2Ptr signedDistanceField() const;

    const ParticleSystemData2Ptr& particleSystemData() const;

 protected:
    void onBeginAdvanceTimeStep(double timeIntervalInSeconds) override;

    void computeAdvection(double timeIntervalInSeconds) override;

    ScalarField2Ptr fluidSdf() const override;

    virtual void transferFromParticlesToGrids();

    virtual void transferFromGridsToParticles();

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
