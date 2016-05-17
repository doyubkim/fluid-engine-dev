// Copyright (c) 2016 Doyub Kim

#ifndef INCLUDE_JET_FMM_LEVEL_SET_SOLVER2_H_
#define INCLUDE_JET_FMM_LEVEL_SET_SOLVER2_H_

#include <jet/level_set_solver2.h>
#include <memory>

namespace jet {

class FmmLevelSetSolver2 final : public LevelSetSolver2 {
 public:
    FmmLevelSetSolver2();

    void reinitialize(
        const ScalarGrid2& inputSdf,
        double maxDistance,
        ScalarGrid2* outputSdf) override;

    void extrapolate(
        const ScalarGrid2& input,
        const ScalarField2& sdf,
        double maxDistance,
        ScalarGrid2* output) override;

    void extrapolate(
        const CollocatedVectorGrid2& input,
        const ScalarField2& sdf,
        double maxDistance,
        CollocatedVectorGrid2* output) override;

    void extrapolate(
        const FaceCenteredGrid2& input,
        const ScalarField2& sdf,
        double maxDistance,
        FaceCenteredGrid2* output) override;

 protected:
    void extrapolate(
        const ConstArrayAccessor2<double>& input,
        const ConstArrayAccessor2<double>& sdf,
        const Vector2D& gridSpacing,
        double maxDistance,
        ArrayAccessor2<double> output);
};

typedef std::shared_ptr<FmmLevelSetSolver2> FmmLevelSetSolver2Ptr;

}  // namespace jet

#endif  // INCLUDE_JET_FMM_LEVEL_SET_SOLVER2_H_
