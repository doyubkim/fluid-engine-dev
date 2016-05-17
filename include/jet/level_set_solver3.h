// Copyright (c) 2016 Doyub Kim

#ifndef INCLUDE_JET_LEVEL_SET_SOLVER3_H_
#define INCLUDE_JET_LEVEL_SET_SOLVER3_H_

#include <jet/collocated_vector_grid3.h>
#include <jet/face_centered_grid3.h>
#include <jet/scalar_grid3.h>
#include <memory>

namespace jet {

class LevelSetSolver3 {
 public:
    LevelSetSolver3();

    virtual ~LevelSetSolver3();

    virtual void reinitialize(
        const ScalarGrid3& inputSdf,
        double maxDistance,
        ScalarGrid3* outputSdf) = 0;

    virtual void extrapolate(
        const ScalarGrid3& input,
        const ScalarField3& sdf,
        double maxDistance,
        ScalarGrid3* output) = 0;

    virtual void extrapolate(
        const CollocatedVectorGrid3& input,
        const ScalarField3& sdf,
        double maxDistance,
        CollocatedVectorGrid3* output) = 0;

    virtual void extrapolate(
        const FaceCenteredGrid3& input,
        const ScalarField3& sdf,
        double maxDistance,
        FaceCenteredGrid3* output) = 0;
};

typedef std::shared_ptr<LevelSetSolver3> LevelSetSolver3Ptr;

}  // namespace jet

#endif  // INCLUDE_JET_LEVEL_SET_SOLVER3_H_
