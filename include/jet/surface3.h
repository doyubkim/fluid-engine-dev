// Copyright (c) 2016 Doyub Kim

#ifndef INCLUDE_JET_SURFACE3_H_
#define INCLUDE_JET_SURFACE3_H_

#include <jet/bounding_box3.h>
#include <jet/geometry3.h>
#include <jet/ray3.h>

namespace jet {

struct SurfaceRayIntersection3 {
    bool isIntersecting = false;
    double t = kMaxD;
    Vector3D point;
    Vector3D normal;
};

class Surface3 : public Geometry3 {
 public:
    Surface3();

    virtual ~Surface3();

    virtual Vector3D closestPoint(const Vector3D& otherPoint) const = 0;

    virtual void getClosestIntersection(
        const Ray3D& ray,
        SurfaceRayIntersection3* intersection) const = 0;

    virtual BoundingBox3D boundingBox() const = 0;

    virtual Vector3D actualClosestNormal(const Vector3D& otherPoint) const = 0;

    virtual bool intersects(const Ray3D& ray) const;

    virtual double closestDistance(const Vector3D& otherPoint) const;

    Vector3D closestNormal(const Vector3D& otherPoint) const;

    void setIsNormalFlipped(bool isFlipped);

    bool isNormalFlipped() const;

 private:
    bool _isNormalFlipped = false;
};

typedef std::shared_ptr<Surface3> Surface3Ptr;

}  // namespace jet

#endif  // INCLUDE_JET_SURFACE3_H_
