// Copyright (c) 2016 Doyub Kim

#ifndef INCLUDE_JET_SURFACE_SET3_H_
#define INCLUDE_JET_SURFACE_SET3_H_

#include <jet/surface3.h>
#include <vector>

namespace jet {

class SurfaceSet3 final : public Surface3 {
 public:
    SurfaceSet3();

    size_t numberOfSurfaces() const;

    const Surface3Ptr& surfaceAt(size_t i) const;

    void addSurface(const Surface3Ptr& surface);

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
    std::vector<Surface3Ptr> _surfaces;
};

typedef std::shared_ptr<SurfaceSet3> SurfaceSet3Ptr;

}  // namespace jet

#endif  // INCLUDE_JET_SURFACE_SET3_H_
