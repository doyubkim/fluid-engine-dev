// Copyright (c) 2016 Doyub Kim

#ifndef INCLUDE_JET_LEVEL_SET_SOLVER2_H_
#define INCLUDE_JET_LEVEL_SET_SOLVER2_H_

#include <jet/collocated_vector_grid2.h>
#include <jet/face_centered_grid2.h>
#include <jet/scalar_grid2.h>
#include <memory>

namespace jet {

class LevelSetSolver2 {
 public:
    LevelSetSolver2();

    virtual ~LevelSetSolver2();

    virtual void reinitialize(
        const ScalarGrid2& inputSdf,
        double maxDistance,
        ScalarGrid2* outputSdf) = 0;

    virtual void extrapolate(
        const ScalarGrid2& input,
        const ScalarField2& sdf,
        double maxDistance,
        ScalarGrid2* output) = 0;

    virtual void extrapolate(
        const CollocatedVectorGrid2& input,
        const ScalarField2& sdf,
        double maxDistance,
        CollocatedVectorGrid2* output) = 0;

    virtual void extrapolate(
        const FaceCenteredGrid2& input,
        const ScalarField2& sdf,
        double maxDistance,
        FaceCenteredGrid2* output) = 0;
};

typedef std::shared_ptr<LevelSetSolver2> LevelSetSolver2Ptr;

}  // namespace jet

#endif  // INCLUDE_JET_LEVEL_SET_SOLVER2_H_
