// Copyright (c) 2018 Doyub Kim
//
// I am making my contributions/submissions to this project solely in my
// personal capacity and am not conveying any rights to any intellectual
// property of any third parties.

#include <pch.h>
#include <jet/point_generator3.h>

namespace jet {

PointGenerator3::PointGenerator3() {
}

PointGenerator3::~PointGenerator3() {
}

void PointGenerator3::generate(
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
