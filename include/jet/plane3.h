// Copyright (c) 2016 Doyub Kim

#ifndef INCLUDE_JET_PLANE3_H_
#define INCLUDE_JET_PLANE3_H_

#include <jet/surface3.h>

namespace jet {

class Plane3 final : public Surface3 {
 public:
    Plane3();

    Plane3(const Vector3D& normal, const Vector3D& point);

    Plane3(
        const Vector3D& point0,
        const Vector3D& point1,
        const Vector3D& point2);

    Plane3(const Plane3& other);

    virtual ~Plane3();

    const Vector3D& normal() const;

    void setNormal(const Vector3D& normal);

    const Vector3D& point() const;

    void setPoint(const Vector3D& point);

    // Surface3 implementation
    Vector3D closestPoint(const Vector3D& otherPoint) const override;

    double closestDistance(const Vector3D& otherPoint) const override;

    Vector3D actualClosestNormal(const Vector3D& otherPoint) const override;

    bool intersects(const Ray3D& ray) const override;

    void getClosestIntersection(
        const Ray3D& ray,
        SurfaceRayIntersection3* intersection) const override;

    BoundingBox3D boundingBox() const override;

 private:
    Vector3D _normal;
    Vector3D _point;
};

typedef std::shared_ptr<Plane3> Plane3Ptr;

}  // namespace jet

#endif  // INCLUDE_JET_PLANE3_H_
