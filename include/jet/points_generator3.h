// Copyright (c) 2016 Doyub Kim

#ifndef INCLUDE_JET_POINTS_GENERATOR3_H_
#define INCLUDE_JET_POINTS_GENERATOR3_H_

#include <jet/array1.h>
#include <jet/bounding_box3.h>

#include <functional>
#include <memory>

namespace jet {

class PointsGenerator3 {
 public:
    PointsGenerator3();

    virtual ~PointsGenerator3();

    virtual void generate(
        const BoundingBox3D& boundingBox,
        double spacing,
        Array1<Vector3D>* points) const;

    virtual void forEachPoint(
        const BoundingBox3D& boundingBox,
        double spacing,
        const std::function<bool(const Vector3D&)>& callback) const = 0;
};

typedef std::shared_ptr<PointsGenerator3> PointsGenerator3Ptr;

}  // namespace jet

#endif  // INCLUDE_JET_POINTS_GENERATOR3_H_
