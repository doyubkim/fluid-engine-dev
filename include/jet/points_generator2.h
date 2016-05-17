// Copyright (c) 2016 Doyub Kim

#ifndef INCLUDE_JET_POINTS_GENERATOR2_H_
#define INCLUDE_JET_POINTS_GENERATOR2_H_

#include <jet/array1.h>
#include <jet/bounding_box2.h>

#include <functional>
#include <memory>

namespace jet {

class PointsGenerator2 {
 public:
    PointsGenerator2();

    virtual ~PointsGenerator2();

    virtual void generate(
        const BoundingBox2D& boundingBox,
        double spacing,
        Array1<Vector2D>* points) const;

    virtual void forEachPoint(
        const BoundingBox2D& boundingBox,
        double spacing,
        const std::function<bool(const Vector2D&)>& callback) const = 0;
};

typedef std::shared_ptr<PointsGenerator2> PointsGenerator2Ptr;

}  // namespace jet

#endif  // INCLUDE_JET_POINTS_GENERATOR2_H_
