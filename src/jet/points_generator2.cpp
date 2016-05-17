// Copyright (c) 2016 Doyub Kim

#include <pch.h>
#include <jet/points_generator2.h>

namespace jet {

PointsGenerator2::PointsGenerator2() {
}

PointsGenerator2::~PointsGenerator2() {
}

void PointsGenerator2::generate(
    const BoundingBox2D& boundingBox,
    double spacing,
    Array1<Vector2D>* points) const {
    forEachPoint(
        boundingBox,
        spacing,
        [&points](const Vector2D& point) {
            points->append(point);
            return true;
        });
}

}  // namespace jet
