// Copyright (c) 2018 Doyub Kim
//
// I am making my contributions/submissions to this project solely in my
// personal capacity and am not conveying any rights to any intellectual
// property of any third parties.

#include <pch.h>
#include <jet/fcc_lattice_point_generator.h>

namespace jet {

void FccLatticePointGenerator::forEachPoint(
    const BoundingBox3D& boundingBox,
    double spacing,
    const std::function<bool(const Vector3D&)>& callback) const {
    double halfSpacing = spacing / 2.0;
    double boxWidth = boundingBox.width();
    double boxHeight = boundingBox.height();
    double boxDepth = boundingBox.depth();

    Vector3D position;
    bool shouldQuit = false;
    for (int k = 0; k * halfSpacing <= boxDepth && !shouldQuit; ++k) {
        position.z = k * halfSpacing + boundingBox.lowerCorner.z;

        bool hasOffset = k % 2 == 1;

        for (int j = 0; j * halfSpacing <= boxHeight && !shouldQuit; ++j) {
            position.y = j * halfSpacing + boundingBox.lowerCorner.y;

            double offset = (hasOffset) ? halfSpacing : 0.0;

            for (int i = 0; i * spacing + offset <= boxWidth; ++i) {
                position.x = i * spacing + offset + boundingBox.lowerCorner.x;
                if (!callback(position)) {
                    shouldQuit = true;
                    break;
                }
            }

            hasOffset = !hasOffset;
        }
    }
}

}  // namespace jet
