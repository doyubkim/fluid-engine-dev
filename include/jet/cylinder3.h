// Copyright (c) 2016 Doyub Kim

#ifndef INCLUDE_JET_CYLINDER3_H_
#define INCLUDE_JET_CYLINDER3_H_

#include <jet/surface3.h>

namespace jet {

class Cylinder3 final : public Surface3 {
 public:
    Cylinder3();

    explicit Cylinder3(const Vector3D& center, double radius, double height);

    Cylinder3(const Cylinder3& other);

    Vector3D closestPoint(const Vector3D& otherPoint) const override;

    double closestDistance(const Vector3D& otherPoint) const override;

    Vector3D actualClosestNormal(const Vector3D& otherPoint) const override;

    bool intersects(const Ray3D& ray) const override;

    void getClosestIntersection(
        const Ray3D& ray,
        SurfaceRayIntersection3* intersection) const override;

    BoundingBox3D boundingBox() const override;

    const Vector3D& center() const;

    void setCenter(const Vector3D& newCenter);

    double radius() const;

    void setRadius(double newRadius);

    double height() const;

    void setHeight(double newHeight);

 private:
    Vector3D _center;
    double _radius = 1.0;
    double _height = 1.0;
};

typedef std::shared_ptr<Cylinder3> Cylinder3Ptr;

}  // namespace jet


#endif  // INCLUDE_JET_CYLINDER3_H_
