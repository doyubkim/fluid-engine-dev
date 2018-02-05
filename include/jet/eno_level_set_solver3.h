// Copyright (c) 2018 Doyub Kim
//
// I am making my contributions/submissions to this project solely in my
// personal capacity and am not conveying any rights to any intellectual
// property of any third parties.

#ifndef INCLUDE_JET_ENO_LEVEL_SET_SOLVER3_H_
#define INCLUDE_JET_ENO_LEVEL_SET_SOLVER3_H_

#include <jet/iterative_level_set_solver3.h>

namespace jet {

//! Three-dimensional third-order ENO-based iterative level set solver.
class EnoLevelSetSolver3 final : public IterativeLevelSetSolver3 {
 public:
    //! Default constructor.
    EnoLevelSetSolver3();

 protected:
    //! Computes the derivatives for given grid point.
    void getDerivatives(ConstArrayAccessor3<double> grid,
                        const Vector3D& gridSpacing, size_t i, size_t j,
                        size_t k, std::array<double, 2>* dx,
                        std::array<double, 2>* dy,
                        std::array<double, 2>* dz) const override;
};

typedef std::shared_ptr<EnoLevelSetSolver3> EnoLevelSetSolver3Ptr;

}  // namespace jet

#endif  // INCLUDE_JET_ENO_LEVEL_SET_SOLVER3_H_
