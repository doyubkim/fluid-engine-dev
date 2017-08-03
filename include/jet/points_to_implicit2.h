// Copyright (c) 2017 Doyub Kim
//
// I am making my contributions/submissions to this project solely in my
// personal capacity and am not conveying any rights to any intellectual
// property of any third parties.

#ifndef INCLUDE_JET_POINTS_TO_IMPLICIT2_H_
#define INCLUDE_JET_POINTS_TO_IMPLICIT2_H_

#include <jet/array_accessor1.h>
#include <jet/scalar_grid2.h>
#include <jet/vector2.h>

namespace jet {

class PointsToImplicit2 {
 public:
    PointsToImplicit2();
    virtual ~PointsToImplicit2();

    virtual void convert(const ConstArrayAccessor1<Vector2D>& points,
                         ScalarGrid2* output) const = 0;
};

}  // namespace jet

#endif  // INCLUDE_JET_POINTS_TO_IMPLICIT2_H_
