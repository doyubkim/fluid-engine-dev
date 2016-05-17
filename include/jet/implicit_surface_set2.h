// Copyright (c) 2016 Doyub Kim

#ifndef INCLUDE_JET_IMPLICIT_SURFACE_SET2_H_
#define INCLUDE_JET_IMPLICIT_SURFACE_SET2_H_

#include <jet/implicit_surface2.h>
#include <vector>

namespace jet {

class ImplicitSurfaceSet2 final : public ImplicitSurface2 {
 public:
    ImplicitSurfaceSet2();

    size_t numberOfSurfaces() const;

    const ImplicitSurface2Ptr& surfaceAt(size_t i) const;

    void addSurface(const Surface2Ptr& surface);

    void addImplicitSurface(const ImplicitSurface2Ptr& surface);

    // Surface2 implementation
    Vector2D closestPoint(const Vector2D& otherPoint) const override;

    double closestDistance(const Vector2D& otherPoint) const override;

    Vector2D actualClosestNormal(const Vector2D& otherPoint) const override;

    bool intersects(const Ray2D& ray) const override;

    void getClosestIntersection(
        const Ray2D& ray,
        SurfaceRayIntersection2* intersection) const override;

    BoundingBox2D boundingBox() const override;

    // ImplicitSurface2 implementation
    double signedDistance(const Vector2D& otherPoint) const override;

 private:
    std::vector<ImplicitSurface2Ptr> _surfaces;
};

typedef std::shared_ptr<ImplicitSurfaceSet2> ImplicitSurfaceSet2Ptr;

}  // namespace jet

#endif  // INCLUDE_JET_IMPLICIT_SURFACE_SET2_H_
