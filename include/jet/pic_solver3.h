// Copyright (c) 2016 Doyub Kim

#ifndef INCLUDE_JET_PIC_SOLVER3_H_
#define INCLUDE_JET_PIC_SOLVER3_H_

#include <jet/grid_fluid_solver3.h>
#include <jet/particle_system_data3.h>

namespace jet {

class PicSolver3 : public GridFluidSolver3 {
 public:
    PicSolver3();

    virtual ~PicSolver3();

    ScalarGrid3Ptr signedDistanceField() const;

    const ParticleSystemData3Ptr& particleSystemData() const;

 protected:
    void onBeginAdvanceTimeStep(double timeIntervalInSeconds) override;

    void computeAdvection(double timeIntervalInSeconds) override;

    ScalarField3Ptr fluidSdf() const override;

    virtual void transferFromParticlesToGrids();

    virtual void transferFromGridsToParticles();

    virtual void moveParticles(double timeIntervalInSeconds);

 private:
    size_t _signedDistanceFieldId;
    ParticleSystemData3Ptr _particles;

    Array3<char> _uMarkers;
    Array3<char> _vMarkers;
    Array3<char> _wMarkers;

    void extrapolateVelocityToAir();

    void buildSignedDistanceField();
};

}  // namespace jet

#endif  // INCLUDE_JET_PIC_SOLVER3_H_
