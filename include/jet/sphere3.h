// Copyright (c) 2016 Doyub Kim

#ifndef INCLUDE_JET_SPHERE3_H_
#define INCLUDE_JET_SPHERE3_H_

#include <jet/surface3.h>
#include <jet/bounding_box3.h>

namespace jet {

class Sphere3 final : public Surface3 {
 public:
    Sphere3(const Vector3D& center, double radius);

    Vector3D closestPoint(const Vector3D& otherPoint) const override;

    Vector3D actualClosestNormal(const Vector3D& otherPoint) const override;

    void getClosestIntersection(
        const Ray3D& ray,
        SurfaceRayIntersection3* intersection) const override;

    BoundingBox3D boundingBox() const override;

    const Vector3D& center() const;

    void setCenter(const Vector3D& newCenter);

    double radius() const;

    void setRadius(double newRadius);

 private:
    Vector3D _center;
    double _radius = 1.0;
};

typedef std::shared_ptr<Sphere3> Sphere3Ptr;

}  // namespace jet


#endif  // INCLUDE_JET_SPHERE3_H_
