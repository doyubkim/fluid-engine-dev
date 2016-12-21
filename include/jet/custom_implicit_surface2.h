// Copyright (c) 2016 Doyub Kim

#ifndef INCLUDE_JET_CUSTOM_IMPLICIT_SURFACE2_H_
#define INCLUDE_JET_CUSTOM_IMPLICIT_SURFACE2_H_

#include <jet/implicit_surface2.h>
#include <jet/scalar_field2.h>

namespace jet {

//! Custom 2-D implicit surface using arbitrary function.
class CustomImplicitSurface2 final : public ImplicitSurface2 {
 public:
    class Builder;

    //! Constructs an implicit surface using the given signed-distance function.
    CustomImplicitSurface2(
        const std::function<double(const Vector2D&)>& func,
        const BoundingBox2D& domain = BoundingBox2D(),
        double resolution = 1e-2,
        bool isNormalFlipped = false);

    //! Destructor.
    virtual ~CustomImplicitSurface2();

    // MARK Surface2 implementations

    //! Returns the closest point from the given point \p otherPoint to the
    //! surface.
    Vector2D closestPoint(const Vector2D& otherPoint) const override;

    //! Returns the closest distance from the given point \p otherPoint to the
    //! point on the surface.
    double closestDistance(const Vector2D& otherPoint) const override;

    //! Returns true if the given \p ray intersects with this object.
    bool intersects(const Ray2D& ray) const override;

    //! Returns the bounding box of this box object.
    BoundingBox2D boundingBox() const override;

    // MARK ImplicitSurface2 implementations

    //! Returns signed distance from the given point \p otherPoint.
    double signedDistance(const Vector2D& otherPoint) const override;

    //! Returns builder for CustomImplicitSurface2.
    static Builder builder();

 private:
    std::function<double(const Vector2D&)> _func;
    BoundingBox2D _domain;
    double _resolution = 1e-2;

    Vector2D actualClosestNormal(const Vector2D& otherPoint) const override;

    SurfaceRayIntersection2 actualClosestIntersection(
        const Ray2D& ray) const override;

    Vector2D gradient(const Vector2D& x) const;
};

//! Shared pointer type for the CustomImplicitSurface2.
typedef std::shared_ptr<CustomImplicitSurface2> CustomImplicitSurface2Ptr;


//!
//! \brief Front-end to create CustomImplicitSurface2 objects step by step.
//!
class CustomImplicitSurface2::Builder final {
 public:
    //! Returns builder with normal direction.
    Builder& withIsNormalFlipped(bool isNormalFlipped);

    //! Returns builder with custom signed-distance function
    Builder& withSignedDistanceFunction(
        const std::function<double(const Vector2D&)>& func);

    //! Returns builder with domain.
    Builder& withDomain(const BoundingBox2D& domain);

    //! Returns builder with resolution.
    Builder& withResolution(double resolution);

    //! Builds CustomImplicitSurface2.
    CustomImplicitSurface2 build() const;

    //! Builds shared pointer of CustomImplicitSurface2 instance.
    CustomImplicitSurface2Ptr makeShared() const;

 private:
    bool _isNormalFlipped = false;
    std::function<double(const Vector2D&)> _func;
    BoundingBox2D _domain;
    double _resolution = 1e-2;
};

}  // namespace jet

#endif  // INCLUDE_JET_CUSTOM_IMPLICIT_SURFACE2_H_
