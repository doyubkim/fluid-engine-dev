// Copyright (c) 2017 Doyub Kim
//
// I am making my contributions/submissions to this project solely in my
// personal capacity and am not conveying any rights to any intellectual
// property of any third parties.

#ifndef INCLUDE_JET_SPH_POINTS_TO_IMPLICIT2_H_
#define INCLUDE_JET_SPH_POINTS_TO_IMPLICIT2_H_

#include <jet/points_to_implicit2.h>

namespace jet {

class SphPointsToImplicit2 final : public PointsToImplicit2 {
 public:
    SphPointsToImplicit2(double kernelRadius = 1.0);

    void convert(const ConstArrayAccessor1<Vector2D>& points,
                 ScalarGrid2* output) const override;

 private:
    double _kernelRadius = 1.0;
};

}  // namespace jet

#endif  // INCLUDE_JET_SPH_POINTS_TO_IMPLICIT2_H_
