// Copyright (c) 2016 Doyub Kim

#ifndef INCLUDE_JET_UPWIND_LEVEL_SET_SOLVER3_H_
#define INCLUDE_JET_UPWIND_LEVEL_SET_SOLVER3_H_

#include <jet/iterative_level_set_solver3.h>

namespace jet {

class UpwindLevelSetSolver3 final : public IterativeLevelSetSolver3 {
 public:
    UpwindLevelSetSolver3();

 protected:
    void getDerivatives(
        ConstArrayAccessor3<double> grid,
        const Vector3D& gridSpacing,
        size_t i,
        size_t j,
        size_t k,
        std::array<double, 2>* dx,
        std::array<double, 2>* dy,
        std::array<double, 2>* dz) const override;
};

}  // namespace jet

#endif  // INCLUDE_JET_UPWIND_LEVEL_SET_SOLVER3_H_
