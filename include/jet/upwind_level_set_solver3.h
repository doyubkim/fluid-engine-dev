// Copyright (c) 2016 Doyub Kim

#ifndef INCLUDE_JET_UPWIND_LEVEL_SET_SOLVER3_H_
#define INCLUDE_JET_UPWIND_LEVEL_SET_SOLVER3_H_

#include <jet/iterative_level_set_solver3.h>

namespace jet {

//! Three-dimensional first-order upwind-based iterative level set solver.
class UpwindLevelSetSolver3 final : public IterativeLevelSetSolver3 {
 public:
    //! Default constructor.
    UpwindLevelSetSolver3();

 protected:
    //! Computes the derivatives for given grid point.
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
