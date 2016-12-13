// Copyright (c) 2016 Doyub Kim

#ifndef INCLUDE_JET_CUSTOM_IMPLICIT_SURFACE3_H_
#define INCLUDE_JET_CUSTOM_IMPLICIT_SURFACE3_H_

#include <jet/implicit_surface3.h>
#include <jet/scalar_field3.h>

namespace jet {

//! Custom 3-D implicit surface using arbitrary function.
class CustomImplicitSurface3 final : public ImplicitSurface3 {
 public:
    class Builder;

    //! Constructs an implicit surface using the given signed-distance function.
    CustomImplicitSurface3(
        const std::function<double(const Vector3D&)>& func,
        const BoundingBox3D& domain = BoundingBox3D(),
        double resolution = 1e-3,
        bool isNormalFlipped = false);

    //! Destructor.
    virtual ~CustomImplicitSurface3();

    // MARK Surface3 implementations

    //! Returns the closest point from the given point \p otherPoint to the
    //! surface.
    Vector3D closestPoint(const Vector3D& otherPoint) const override;

    //! Returns the closest distance from the given point \p otherPoint to the
    //! point on the surface.
    double closestDistance(const Vector3D& otherPoint) const override;

    //! Returns true if the given \p ray intersects with this object.
    bool intersects(const Ray3D& ray) const override;

    //! Returns the bounding box of this box object.
    BoundingBox3D boundingBox() const override;

    // MARK ImplicitSurface3 implementations

    //! Returns signed distance from the given point \p otherPoint.
    double signedDistance(const Vector3D& otherPoint) const override;

    //! Returns builder for CustomImplicitSurface3.
    static Builder builder();

 private:
    std::function<double(const Vector3D&)> _func;
    BoundingBox3D _domain;
    double _resolution = 1e-3;

    Vector3D actualClosestNormal(const Vector3D& otherPoint) const override;

    SurfaceRayIntersection3 actualClosestIntersection(
        const Ray3D& ray) const override;

    Vector3D gradient(const Vector3D& x) const;
};

//! Shared pointer type for the CustomImplicitSurface3.
typedef std::shared_ptr<CustomImplicitSurface3> CustomImplicitSurface3Ptr;


//!
//! \brief Front-end to create CustomImplicitSurface3 objects step by step.
//!
class CustomImplicitSurface3::Builder final {
 public:
    //! Returns builder with normal direction.
    Builder& withIsNormalFlipped(bool isNormalFlipped);

    //! Returns builder with custom signed-distance function
    Builder& withSignedDistanceFunction(
        const std::function<double(const Vector3D&)>& func);

    //! Returns builder with domain.
    Builder& withDomain(const BoundingBox3D& domain);

    //! Returns builder with resolution.
    Builder& withResolution(double resolution);

    //! Builds CustomImplicitSurface3.
    CustomImplicitSurface3 build() const;

    //! Builds shared pointer of CustomImplicitSurface3 instance.
    CustomImplicitSurface3Ptr makeShared() const {
        return std::make_shared<CustomImplicitSurface3>(
            _func,
            _domain,
            _resolution,
            _isNormalFlipped);
    }

 private:
    bool _isNormalFlipped = false;
    std::function<double(const Vector3D&)> _func;
    BoundingBox3D _domain;
    double _resolution = 1e-3;
};

}  // namespace jet

#endif  // INCLUDE_JET_CUSTOM_IMPLICIT_SURFACE3_H_
