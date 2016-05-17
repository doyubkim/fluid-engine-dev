// Copyright (c) 2016 Doyub Kim

#include <pch.h>
#include <jet/points_generator3.h>

namespace jet {

PointsGenerator3::PointsGenerator3() {
}

PointsGenerator3::~PointsGenerator3() {
}

void PointsGenerator3::generate(
    const BoundingBox3D& boundingBox,
    double spacing,
    Array1<Vector3D>* points) const {
    forEachPoint(
        boundingBox,
        spacing,
        [&points](const Vector3D& point) {
            points->append(point);
            return true;
        });
}

}  // namespace jet
