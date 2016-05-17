// Copyright (c) 2016 Doyub Kim

#ifndef INCLUDE_JET_SURFACE_TO_IMPLICIT3_H_
#define INCLUDE_JET_SURFACE_TO_IMPLICIT3_H_

#include <jet/implicit_surface3.h>
#include <memory>

namespace jet {

class SurfaceToImplicit3 final : public ImplicitSurface3 {
 public:
    explicit SurfaceToImplicit3(const Surface3Ptr& surface);

    // Surface3 implementation
    Vector3D closestPoint(const Vector3D& otherPoint) const override;

    Vector3D actualClosestNormal(const Vector3D& otherPoint) const override;

    double closestDistance(const Vector3D& otherPoint) const override;

    bool intersects(const Ray3D& ray) const override;

    void getClosestIntersection(
        const Ray3D& ray,
        SurfaceRayIntersection3* intersection) const override;

    BoundingBox3D boundingBox() const override;

    // ImplicitSurface3 implementation
    double signedDistance(const Vector3D& otherPoint) const override;

 private:
    Surface3Ptr _surface;
};

typedef std::shared_ptr<SurfaceToImplicit3> SurfaceToImplicit3Ptr;

}  // namespace jet

#endif  // INCLUDE_JET_SURFACE_TO_IMPLICIT3_H_
