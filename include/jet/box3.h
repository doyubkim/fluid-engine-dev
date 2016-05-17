// Copyright (c) 2016 Doyub Kim

#ifndef INCLUDE_JET_BOX3_H_
#define INCLUDE_JET_BOX3_H_

#include <jet/surface3.h>
#include <jet/bounding_box3.h>

namespace jet {

class Box3 final : public Surface3 {
 public:
    Box3();

    explicit Box3(const Vector3D& lowerCorner, const Vector3D& upperCorner);

    explicit Box3(const BoundingBox3D& boundingBox);

    Box3(const Box3& other);

    // Surface3 implementation
    Vector3D closestPoint(const Vector3D& otherPoint) const override;

    Vector3D actualClosestNormal(const Vector3D& otherPoint) const override;

    double closestDistance(const Vector3D& otherPoint) const override;

    bool intersects(const Ray3D& ray) const override;

    void getClosestIntersection(
        const Ray3D& ray,
        SurfaceRayIntersection3* intersection) const override;

    BoundingBox3D boundingBox() const override;

 private:
    BoundingBox3D _boundingBox
        = BoundingBox3D(Vector3D(), Vector3D(1.0, 1.0, 1.0));
};

typedef std::shared_ptr<Box3> Box3Ptr;

}  // namespace jet


#endif  // INCLUDE_JET_BOX3_H_
