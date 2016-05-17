// Copyright (c) 2016 Doyub Kim

#ifndef INCLUDE_JET_IMPLICIT_SURFACE_SET3_H_
#define INCLUDE_JET_IMPLICIT_SURFACE_SET3_H_

#include <jet/implicit_surface3.h>
#include <vector>

namespace jet {

class ImplicitSurfaceSet3 final : public ImplicitSurface3 {
 public:
    ImplicitSurfaceSet3();

    size_t numberOfSurfaces() const;

    const ImplicitSurface3Ptr& surfaceAt(size_t i) const;

    void addSurface(const Surface3Ptr& surface);

    void addImplicitSurface(const ImplicitSurface3Ptr& surface);

    // Surface3 implementation
    Vector3D closestPoint(const Vector3D& otherPoint) const override;

    double closestDistance(const Vector3D& otherPoint) const override;

    Vector3D actualClosestNormal(const Vector3D& otherPoint) const override;

    bool intersects(const Ray3D& ray) const override;

    void getClosestIntersection(
        const Ray3D& ray,
        SurfaceRayIntersection3* intersection) const override;

    BoundingBox3D boundingBox() const override;

    // ImplicitSurface3 implementation
    double signedDistance(const Vector3D& otherPoint) const override;

 private:
    std::vector<ImplicitSurface3Ptr> _surfaces;
};

typedef std::shared_ptr<ImplicitSurfaceSet3> ImplicitSurfaceSet3Ptr;

}  // namespace jet

#endif  // INCLUDE_JET_IMPLICIT_SURFACE_SET3_H_
