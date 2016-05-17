// Copyright (c) 2016 Doyub Kim

#ifndef INCLUDE_JET_FLIP_SOLVER3_H_
#define INCLUDE_JET_FLIP_SOLVER3_H_

#include <jet/pic_solver3.h>

namespace jet {

class FlipSolver3 : public PicSolver3 {
 public:
    FlipSolver3();

    virtual ~FlipSolver3();

 protected:
    void transferFromParticlesToGrids() override;

    void transferFromGridsToParticles() override;

 private:
    FaceCenteredGrid3 _delta;
};

}  // namespace jet

#endif  // INCLUDE_JET_FLIP_SOLVER3_H_
