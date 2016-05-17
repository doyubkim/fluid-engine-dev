// Copyright (c) 2016 Doyub Kim

#ifndef INCLUDE_JET_FLIP_SOLVER2_H_
#define INCLUDE_JET_FLIP_SOLVER2_H_

#include <jet/pic_solver2.h>

namespace jet {

class FlipSolver2 : public PicSolver2 {
 public:
    FlipSolver2();

    virtual ~FlipSolver2();

 protected:
    void transferFromParticlesToGrids() override;

    void transferFromGridsToParticles() override;

 private:
    FaceCenteredGrid2 _delta;
};

}  // namespace jet

#endif  // INCLUDE_JET_FLIP_SOLVER2_H_
