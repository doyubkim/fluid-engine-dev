// Copyright (c) 2018 Doyub Kim
//
// I am making my contributions/submissions to this project solely in my
// personal capacity and am not conveying any rights to any intellectual
// property of any third parties.

#ifndef INCLUDE_JET_IMPLICIT_SURFACE_SET_H_
#define INCLUDE_JET_IMPLICIT_SURFACE_SET_H_

#include <jet/array.h>
#include <jet/bvh.h>
#include <jet/implicit_surface.h>

namespace jet {

//!
//! \brief N-D implicit surface set.
//!
//! This class represents N-D implicit surface set which extends
//! ImplicitSurface by overriding implicit surface-related queries. This is
//! class can hold a collection of other implicit surface instances.
//!
template <size_t N>
class ImplicitSurfaceSet final : public ImplicitSurface<N> {
 public:
    class Builder;

    using ImplicitSurface<N>::transform;
    using ImplicitSurface<N>::isNormalFlipped;

    //! Constructs an empty implicit surface set.
    ImplicitSurfaceSet();

    //! Constructs an implicit surface set using list of other surfaces.
    ImplicitSurfaceSet(
        ConstArrayView1<std::shared_ptr<ImplicitSurface<N>>> surfaces,
        const Transform<N>& transform = Transform<N>(),
        bool isNormalFlipped = false);

    //! Constructs an implicit surface set using list of other surfaces.
    ImplicitSurfaceSet(
        ConstArrayView1<std::shared_ptr<Surface<N>>> surfaces,
        const Transform<N>& transform = Transform<N>(),
        bool isNormalFlipped = false);

    //! Copy constructor.
    ImplicitSurfaceSet(const ImplicitSurfaceSet& other);

    //! Updates internal spatial query engine.
    void updateQueryEngine() override;

    //! Returns true if the surface is a valid geometry.
    bool isValidGeometry() const override;

    //! Returns the number of implicit surfaces.
    size_t numberOfSurfaces() const;

    //! Returns the i-th implicit surface.
    const std::shared_ptr<ImplicitSurface<N>>& surfaceAt(size_t i) const;

    //! Adds an explicit surface instance.
    void addExplicitSurface(const std::shared_ptr<Surface<N>>& surface);

    //! Adds an implicit surface instance.
    void addSurface(const std::shared_ptr<ImplicitSurface<N>>& surface);

    //! Returns builder fox ImplicitSurfaceSet.
    static Builder builder();

 private:
    Array1<std::shared_ptr<ImplicitSurface<N>>> _surfaces;
    Array1<std::shared_ptr<ImplicitSurface<N>>> _unboundedSurfaces;
    mutable Bvh<std::shared_ptr<ImplicitSurface<N>>, N> _bvh;
    mutable bool _bvhInvalidated = true;

    // Surface implementations

    Vector<double, N> closestPointLocal(
        const Vector<double, N>& otherPoint) const override;

    BoundingBox<double, N> boundingBoxLocal() const override;

    double closestDistanceLocal(
        const Vector<double, N>& otherPoint) const override;

    bool intersectsLocal(const Ray<double, N>& ray) const override;

    Vector<double, N> closestNormalLocal(
        const Vector<double, N>& otherPoint) const override;

    SurfaceRayIntersection<N> closestIntersectionLocal(
        const Ray<double, N>& ray) const override;

    // ImplicitSurface implementations

    double signedDistanceLocal(
        const Vector<double, N>& otherPoint) const override;

    void invalidateBvh();

    void buildBvh() const;
};

//! 2-D ImplicitSurfaceSet type.
using ImplicitSurfaceSet2 = ImplicitSurfaceSet<2>;

//! 3-D ImplicitSurfaceSet type.
using ImplicitSurfaceSet3 = ImplicitSurfaceSet<3>;

//! Shared pointer type for the ImplicitSurfaceSet2.
using ImplicitSurfaceSet2Ptr = std::shared_ptr<ImplicitSurfaceSet2>;

//! Shared pointer type for the ImplicitSurfaceSet3.
using ImplicitSurfaceSet3Ptr = std::shared_ptr<ImplicitSurfaceSet3>;

//!
//! \brief Front-end to create ImplicitSurfaceSet objects step by step.
//!
template <size_t N>
class ImplicitSurfaceSet<N>::Builder final
    : public SurfaceBuilderBase<N, ImplicitSurfaceSet<N>::Builder> {
 public:
    //! Returns builder with surfaces.
    Builder& withSurfaces(
        const ConstArrayView1<std::shared_ptr<ImplicitSurface<N>>>& surfaces);

    //! Returns builder with explicit surfaces.
    Builder& withExplicitSurfaces(
        const ConstArrayView1<std::shared_ptr<Surface<N>>>& surfaces);

    //! Builds ImplicitSurfaceSet.
    ImplicitSurfaceSet<N> build() const;

    //! Builds shared pointer of ImplicitSurfaceSet instance.
    std::shared_ptr<ImplicitSurfaceSet<N>> makeShared() const;

 private:
    using Base = SurfaceBuilderBase<N, ImplicitSurfaceSet<N>::Builder>;
    using Base::_transform;
    using Base::_isNormalFlipped;

    Array1<std::shared_ptr<ImplicitSurface<N>>> _surfaces;
};

}  // namespace jet

#endif  // INCLUDE_JET_IMPLICIT_SURFACE_SET_H_
