// Copyright (c) 2016 Doyub Kim

#ifndef INCLUDE_JET_PLANE2_H_
#define INCLUDE_JET_PLANE2_H_

#include <jet/surface2.h>

namespace jet {

class Plane2 final : public Surface2 {
 public:
    Plane2();

    Plane2(const Vector2D& normal, const Vector2D& point);

    Plane2(
        const Vector2D& point0,
        const Vector2D& point1,
        const Vector2D& point2);

    Plane2(const Plane2& other);

    virtual ~Plane2();

    const Vector2D& normal() const;

    void setNormal(const Vector2D& normal);

    const Vector2D& point() const;

    void setPoint(const Vector2D& point);

    // Surface2 implementation
    Vector2D closestPoint(const Vector2D& otherPoint) const override;

    double closestDistance(const Vector2D& otherPoint) const override;

    Vector2D actualClosestNormal(const Vector2D& otherPoint) const override;

    bool intersects(const Ray2D& ray) const override;

    void getClosestIntersection(
        const Ray2D& ray,
        SurfaceRayIntersection2* intersection) const override;

    BoundingBox2D boundingBox() const override;

 private:
    Vector2D _normal;
    Vector2D _point;
};

typedef std::shared_ptr<Plane2> Plane2Ptr;

}  // namespace jet

#endif  // INCLUDE_JET_PLANE2_H_
