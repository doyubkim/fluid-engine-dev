// Copyright (c) 2018 Doyub Kim
//
// I am making my contributions/submissions to this project solely in my
// personal capacity and am not conveying any rights to any intellectual
// property of any third parties.

#include <pch.h>
#include <jet/grid_point_generator3.h>

namespace jet {

void GridPointGenerator3::forEachPoint(
    const BoundingBox3D& boundingBox,
    double spacing,
    const std::function<bool(const Vector3D&)>& callback) const {
    Vector3D position;
    double boxWidth = boundingBox.width();
    double boxHeight = boundingBox.height();
    double boxDepth = boundingBox.depth();

    bool shouldQuit = false;
    for (int k = 0; k * spacing <= boxDepth && !shouldQuit; ++k) {
        position.z = k * spacing + boundingBox.lowerCorner.z;

        for (int j = 0; j * spacing <= boxHeight && !shouldQuit; ++j) {
            position.y = j * spacing + boundingBox.lowerCorner.y;

            for (int i = 0; i * spacing <= boxWidth && !shouldQuit; ++i) {
                position.x = i * spacing + boundingBox.lowerCorner.x;
                if (!callback(position)) {
                    shouldQuit = true;
                    break;
                }
            }
        }
    }
}

}  // namespace jet
