// Copyright (c) 2018 Doyub Kim
//
// I am making my contributions/submissions to this project solely in my
// personal capacity and am not conveying any rights to any intellectual
// property of any third parties.

#include <pch.h>
#include <jet/point_generator2.h>

namespace jet {

PointGenerator2::PointGenerator2() {
}

PointGenerator2::~PointGenerator2() {
}

void PointGenerator2::generate(
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
