// Copyright (c) 2018 Doyub Kim
//
// I am making my contributions/submissions to this project solely in my
// personal capacity and am not conveying any rights to any intellectual
// property of any third parties.

#include <pch.h>
#include <jet/bcc_lattice_point_generator.h>

namespace jet {

void BccLatticePointGenerator::forEachPoint(
    const BoundingBox3D& boundingBox,
    double spacing,
    const std::function<bool(const Vector3D&)>& callback) const {
    double halfSpacing = spacing / 2.0;
    double boxWidth = boundingBox.width();
    double boxHeight = boundingBox.height();
    double boxDepth = boundingBox.depth();

    Vector3D position;
    bool hasOffset = false;
    bool shouldQuit = false;
    for (int k = 0; k * halfSpacing <= boxDepth && !shouldQuit; ++k) {
        position.z = k * halfSpacing + boundingBox.lowerCorner.z;

        double offset = (hasOffset) ? halfSpacing : 0.0;

        for (int j = 0; j * spacing + offset <= boxHeight && !shouldQuit; ++j) {
            position.y = j * spacing + offset + boundingBox.lowerCorner.y;

            for (int i = 0; i * spacing + offset <= boxWidth; ++i) {
                position.x = i * spacing + offset + boundingBox.lowerCorner.x;
                if (!callback(position)) {
                    shouldQuit = true;
                    break;
                }
            }
        }

        hasOffset = !hasOffset;
    }
}

}  // namespace jet
