// Copyright (c) 2018 Doyub Kim
//
// I am making my contributions/submissions to this project solely in my
// personal capacity and am not conveying any rights to any intellectual
// property of any third parties.

#ifndef INCLUDE_JET_CUSTOM_IMPLICIT_SURFACE_H_
#define INCLUDE_JET_CUSTOM_IMPLICIT_SURFACE_H_

#include <jet/implicit_surface.h>
#include <jet/scalar_field.h>

namespace jet {

//! Custom N-D implicit surface using arbitrary function.
template <size_t N>
class CustomImplicitSurface final : public ImplicitSurface<N> {
 public:
    class Builder;

    //!
    //! Constructs an implicit surface using the given signed-distance function.
    //!
    //! \param func Custom SDF function object.
    //! \param domain Bounding box of the SDF if exists.
    //! \param resolution Finite differencing resolution for derivatives.
    //! \param rayMarchingResolution Ray marching resolution for ray tests.
    //! \param maxNumOfIterations Number of iterations for closest point search.
    //! \param transform Local-to-world transform.
    //! \param isNormalFlipped True if normal is flipped.
    //!
    CustomImplicitSurface(
        const std::function<double(const Vector<double, N>&)>& func,
        const BoundingBox<double, N>& domain = BoundingBox<double, N>(),
        double resolution = 1e-3, double rayMarchingResolution = 1e-6,
        unsigned int numberOfIterations = 5,
        const Transform<N>& transform = Transform<N>(),
        bool isNormalFlipped = false);

    //! Destructor.
    virtual ~CustomImplicitSurface();

    //! Returns builder for CustomImplicitSurface.
    static Builder builder();

 private:
    std::function<double(const Vector<double, N>&)> _func;
    BoundingBox<double, N> _domain;
    double _resolution = 1e-3;
    double _rayMarchingResolution = 1e-6;
    unsigned int _maxNumOfIterations = 5;

    Vector<double, N> closestPointLocal(
        const Vector<double, N>& otherPoint) const override;

    bool intersectsLocal(const Ray<double, N>& ray) const override;

    BoundingBox<double, N> boundingBoxLocal() const override;

    Vector<double, N> closestNormalLocal(
        const Vector<double, N>& otherPoint) const override;

    double signedDistanceLocal(
        const Vector<double, N>& otherPoint) const override;

    SurfaceRayIntersection<N> closestIntersectionLocal(
        const Ray<double, N>& ray) const override;

    Vector<double, N> gradientLocal(const Vector<double, N>& x) const;
};

//! 2-D CustomImplicitSurface type.
using CustomImplicitSurface2 = CustomImplicitSurface<2>;

//! 3-D CustomImplicitSurface type.
using CustomImplicitSurface3 = CustomImplicitSurface<3>;

//! Shared pointer type for the CustomImplicitSurface2.
using CustomImplicitSurface2Ptr = std::shared_ptr<CustomImplicitSurface2>;

//! Shared pointer type for the CustomImplicitSurface3.
using CustomImplicitSurface3Ptr = std::shared_ptr<CustomImplicitSurface3>;

//!
//! \brief Front-end to create CustomImplicitSurface objects step by step.
//!
template <size_t N>
class CustomImplicitSurface<N>::Builder final
    : public SurfaceBuilderBase<N, typename CustomImplicitSurface<N>::Builder> {
 public:
    //! Returns builder with custom signed-distance function
    Builder& withSignedDistanceFunction(
        const std::function<double(const Vector<double, N>&)>& func);

    //! Returns builder with domain.
    Builder& withDomain(const BoundingBox<double, N>& domain);

    //! Returns builder with finite differencing resolution.
    Builder& withResolution(double resolution);

    //! Returns builder with ray marching resolution which determines the ray
    //! intersection quality.
    Builder& withRayMarchingResolution(double rayMarchingResolution);

    //! Returns builder with number of iterations for closest point/normal
    //! searches.
    Builder& withMaxNumberOfIterations(unsigned int numIter);

    //! Builds CustomImplicitSurface.
    CustomImplicitSurface<N> build() const;

    //! Builds shared pointer of CustomImplicitSurface instance.
    std::shared_ptr<CustomImplicitSurface<N>> makeShared() const;

 private:
    using Base = SurfaceBuilderBase<N, typename CustomImplicitSurface<N>::Builder>;
    using Base::_transform;
    using Base::_isNormalFlipped;

    std::function<double(const Vector<double, N>&)> _func;
    BoundingBox<double, N> _domain;
    double _resolution = 1e-3;
    double _rayMarchingResolution = 1e-6;
    unsigned int _maxNumOfIterations = 5;
};

}  // namespace jet

#endif  // INCLUDE_JET_CUSTOM_IMPLICIT_SURFACE_H_
