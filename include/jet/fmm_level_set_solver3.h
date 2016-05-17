// Copyright (c) 2016 Doyub Kim

#ifndef INCLUDE_JET_FMM_LEVEL_SET_SOLVER3_H_
#define INCLUDE_JET_FMM_LEVEL_SET_SOLVER3_H_

#include <jet/level_set_solver3.h>
#include <memory>

namespace jet {

class FmmLevelSetSolver3 final : public LevelSetSolver3 {
 public:
    FmmLevelSetSolver3();

    void reinitialize(
        const ScalarGrid3& inputSdf,
        double maxDistance,
        ScalarGrid3* outputSdf) override;

    void extrapolate(
        const ScalarGrid3& input,
        const ScalarField3& sdf,
        double maxDistance,
        ScalarGrid3* output) override;

    void extrapolate(
        const CollocatedVectorGrid3& input,
        const ScalarField3& sdf,
        double maxDistance,
        CollocatedVectorGrid3* output) override;

    void extrapolate(
        const FaceCenteredGrid3& input,
        const ScalarField3& sdf,
        double maxDistance,
        FaceCenteredGrid3* output) override;

 protected:
    void extrapolate(
        const ConstArrayAccessor3<double>& input,
        const ConstArrayAccessor3<double>& sdf,
        const Vector3D& gridSpacing,
        double maxDistance,
        ArrayAccessor3<double> output);
};

typedef std::shared_ptr<FmmLevelSetSolver3> FmmLevelSetSolver3Ptr;

}  // namespace jet

#endif  // INCLUDE_JET_FMM_LEVEL_SET_SOLVER3_H_
