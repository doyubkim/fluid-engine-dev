// Copyright (c) 2016 Doyub Kim

#ifndef INCLUDE_JET_GRID_POINTS_GENERATOR3_H_
#define INCLUDE_JET_GRID_POINTS_GENERATOR3_H_

#include <jet/points_generator3.h>

namespace jet {

class GridPointsGenerator3 final : public PointsGenerator3 {
 public:
    void forEachPoint(
        const BoundingBox3D& boundingBox,
        double spacing,
        const std::function<bool(const Vector3D&)>& callback) const;
};

typedef std::shared_ptr<GridPointsGenerator3> GridPointsGenerator3Ptr;

}  // namespace jet

#endif  // INCLUDE_JET_GRID_POINTS_GENERATOR3_H_
