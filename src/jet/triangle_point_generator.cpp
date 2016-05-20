// Copyright (c) 2016 Doyub Kim

#include <pch.h>
#include <jet/triangle_point_generator.h>

using namespace jet;

void TrianglePointGenerator::forEachPoint(
    const BoundingBox2D& boundingBox,
    double spacing,
    const std::function<bool(const Vector2D&)>& callback) const {
    const double halfSpacing = spacing / 2.0;
    const double ySpacing = spacing * std::sqrt(3.0) / 2.0;
    double boxWidth = boundingBox.width();
    double boxHeight = boundingBox.height();

    Vector2D position;
    bool hasOffset = false;
    bool shouldQuit = false;
    for (int j = 0; j * ySpacing <= boxHeight && !shouldQuit; ++j) {
        position.y = j * ySpacing + boundingBox.lowerCorner.y;

        double offset = (hasOffset) ? halfSpacing : 0.0;

        for (int i = 0; i * spacing + offset <= boxWidth && !shouldQuit; ++i) {
            position.x = i * spacing + offset + boundingBox.lowerCorner.x;
            if (!callback(position)) {
                shouldQuit = true;
                break;
            }
        }

        hasOffset = !hasOffset;
    }
}
