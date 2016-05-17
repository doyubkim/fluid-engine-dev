// Copyright (c) 2016 Doyub Kim

#ifndef INCLUDE_JET_SURFACE2_H_
#define INCLUDE_JET_SURFACE2_H_

#include <jet/bounding_box2.h>
#include <jet/geometry2.h>
#include <jet/ray2.h>

namespace jet {

struct SurfaceRayIntersection2 {
    bool isIntersecting;
    double t;
    Vector2D point;
    Vector2D normal;
};

class Surface2 : public Geometry2 {
 public:
    Surface2();

    virtual ~Surface2();

    virtual Vector2D closestPoint(const Vector2D& otherPoint) const = 0;

    virtual void getClosestIntersection(
        const Ray2D& ray,
        SurfaceRayIntersection2* intersection) const = 0;

    virtual BoundingBox2D boundingBox() const = 0;

    virtual Vector2D actualClosestNormal(const Vector2D& otherPoint) const = 0;

    virtual bool intersects(const Ray2D& ray) const;

    virtual double closestDistance(const Vector2D& otherPoint) const;

    Vector2D closestNormal(const Vector2D& otherPoint) const;

    void setIsNormalFlipped(bool isFlipped);

    bool isNormalFlipped() const;

 private:
    bool _isNormalFlipped = false;
};

typedef std::shared_ptr<Surface2> Surface2Ptr;

}  // namespace jet

#endif  // INCLUDE_JET_SURFACE2_H_
