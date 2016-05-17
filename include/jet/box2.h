// Copyright (c) 2016 Doyub Kim

#ifndef INCLUDE_JET_BOX2_H_
#define INCLUDE_JET_BOX2_H_

#include <jet/surface2.h>
#include <jet/bounding_box2.h>

namespace jet {

class Box2 final : public Surface2 {
 public:
    Box2();

    explicit Box2(const Vector2D& lowerCorner, const Vector2D& upperCorner);

    explicit Box2(const BoundingBox2D& boundingBox);

    Box2(const Box2& other);

    // Surface2 implementation
    Vector2D closestPoint(const Vector2D& otherPoint) const override;

    Vector2D actualClosestNormal(const Vector2D& otherPoint) const override;

    double closestDistance(const Vector2D& otherPoint) const override;

    bool intersects(const Ray2D& ray) const override;

    void getClosestIntersection(
        const Ray2D& ray,
        SurfaceRayIntersection2* intersection) const override;

    BoundingBox2D boundingBox() const override;

 private:
    BoundingBox2D _boundingBox
        = BoundingBox2D(Vector2D(), Vector2D(1.0, 1.0));
};

typedef std::shared_ptr<Box2> Box2Ptr;

}  // namespace jet


#endif  // INCLUDE_JET_BOX2_H_
