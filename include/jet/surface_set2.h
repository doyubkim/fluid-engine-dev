// Copyright (c) 2016 Doyub Kim

#ifndef INCLUDE_JET_SURFACE_SET2_H_
#define INCLUDE_JET_SURFACE_SET2_H_

#include <jet/surface2.h>
#include <vector>

namespace jet {

class SurfaceSet2 final : public Surface2 {
 public:
    SurfaceSet2();

    size_t numberOfSurfaces() const;

    const Surface2Ptr& surfaceAt(size_t i) const;

    void addSurface(const Surface2Ptr& surface);

    // Surface2 implementation
    Vector2D closestPoint(const Vector2D& otherPoint) const override;

    Vector2D actualClosestNormal(const Vector2D& otherPoint) const override;

    double closestDistance(const Vector2D& otherPoint) const override;

    bool intersects(const Ray2D& ray) const override;

    void getClosestIntersection(
        const Ray2D& ray,
        SurfaceRayIntersection2* intersection) const override;

    BoundingBox2D boundingBox() const override;

 private:
    std::vector<Surface2Ptr> _surfaces;
};

typedef std::shared_ptr<SurfaceSet2> SurfaceSet2Ptr;

}  // namespace jet

#endif  // INCLUDE_JET_SURFACE_SET2_H_
