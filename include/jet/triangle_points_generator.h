// Copyright (c) 2016 Doyub Kim

#ifndef INCLUDE_JET_TRIANGLE_POINTS_GENERATOR_H_
#define INCLUDE_JET_TRIANGLE_POINTS_GENERATOR_H_

#include <jet/points_generator2.h>

namespace jet {

class TrianglePointsGenerator final : public PointsGenerator2 {
 public:
    void forEachPoint(
        const BoundingBox2D& boundingBox,
        double spacing,
        const std::function<bool(const Vector2D&)>& callback) const override;
};

typedef std::shared_ptr<TrianglePointsGenerator> TrianglePointsGeneratorPtr;

}

#endif  // INCLUDE_JET_TRIANGLE_POINTS_GENERATOR_H_
