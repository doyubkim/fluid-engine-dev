// Copyright (c) 2016 Doyub Kim

#ifndef INCLUDE_JET_SPHERE2_H_
#define INCLUDE_JET_SPHERE2_H_

#include <jet/surface2.h>
#include <jet/bounding_box2.h>

namespace jet {

class Sphere2 final : public Surface2 {
 public:
    Sphere2(const Vector2D& center, double radius);

    Vector2D closestPoint(const Vector2D& otherPoint) const override;

    Vector2D actualClosestNormal(const Vector2D& otherPoint) const override;

    void getClosestIntersection(
        const Ray2D& ray,
        SurfaceRayIntersection2* intersection) const override;

    BoundingBox2D boundingBox() const override;

    const Vector2D& center() const;

    void setCenter(const Vector2D& newCenter);

    double radius() const;

    void setRadius(double newRadius);

 private:
    Vector2D _center;
    double _radius = 1.0;
};

typedef std::shared_ptr<Sphere2> Sphere2Ptr;

}  // namespace jet


#endif  // INCLUDE_JET_SPHERE2_H_
