// Copyright (c) 2016 Doyub Kim

#ifndef INCLUDE_JET_GRID_POINTS_GENERATOR2_H_
#define INCLUDE_JET_GRID_POINTS_GENERATOR2_H_

#include <jet/points_generator2.h>

namespace jet {

class GridPointsGenerator2 final : public PointsGenerator2 {
 public:
    void forEachPoint(
        const BoundingBox2D& boundingBox,
        double spacing,
        const std::function<bool(const Vector2D&)>& callback) const;
};

typedef std::shared_ptr<GridPointsGenerator2> GridPointsGenerator2Ptr;

}  // namespace jet

#endif  // INCLUDE_JET_GRID_POINTS_GENERATOR2_H_
