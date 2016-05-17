// Copyright (c) 2016 Doyub Kim

#ifndef INCLUDE_JET_SURFACE_TO_IMPLICIT2_H_
#define INCLUDE_JET_SURFACE_TO_IMPLICIT2_H_

#include <jet/implicit_surface2.h>
#include <memory>

namespace jet {

class SurfaceToImplicit2 final : public ImplicitSurface2 {
 public:
    explicit SurfaceToImplicit2(const Surface2Ptr& surface);

    // Surface2 implementation
    Vector2D closestPoint(const Vector2D& otherPoint) const override;

    Vector2D actualClosestNormal(const Vector2D& otherPoint) const override;

    double closestDistance(const Vector2D& otherPoint) const override;

    bool intersects(const Ray2D& ray) const override;

    void getClosestIntersection(
        const Ray2D& ray,
        SurfaceRayIntersection2* intersection) const override;

    BoundingBox2D boundingBox() const override;

    // ImplicitSurface2 implementation
    double signedDistance(const Vector2D& otherPoint) const override;

 private:
    Surface2Ptr _surface;
};

typedef std::shared_ptr<SurfaceToImplicit2> SurfaceToImplicit2Ptr;

}  // namespace jet

#endif  // INCLUDE_JET_SURFACE_TO_IMPLICIT2_H_
