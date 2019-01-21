// Copyright (c) 2018 Doyub Kim
//
// I am making my contributions/submissions to this project solely in my
// personal capacity and am not conveying any rights to any intellectual
// property of any third parties.

#ifndef INCLUDE_JET_SURFACE_SET_H_
#define INCLUDE_JET_SURFACE_SET_H_

#include <jet/array.h>
#include <jet/bvh.h>
#include <jet/surface.h>

namespace jet {

//!
//! \brief N-D surface set.
//!
//! This class represents N-D surface set which extends Surface by overriding
//! surface-related queries. This is class can hold a collection of other
//! surface instances.
//!
template <size_t N>
class SurfaceSet final : public Surface<N> {
 public:
    class Builder;

    //! Constructs an empty surface set.
    SurfaceSet();

    //! Constructs with a list of other surfaces.
    explicit SurfaceSet(const ConstArrayView1<std::shared_ptr<Surface<N>>>& others,
                        const Transform<N>& transform = Transform<N>(),
                        bool isNormalFlipped = false);

    //! Copy constructor.
    SurfaceSet(const SurfaceSet& other);

    //! Updates internal spatial query engine.
    void updateQueryEngine() override;

    //! Returns true if bounding box can be defined.
    bool isBounded() const override;

    //! Returns true if the surface is a valid geometry.
    bool isValidGeometry() const override;

    //! Returns the number of surfaces.
    size_t numberOfSurfaces() const;

    //! Returns the i-th surface.
    const std::shared_ptr<Surface<N>>& surfaceAt(size_t i) const;

    //! Adds a surface instance.
    void addSurface(const std::shared_ptr<Surface<N>>& surface);

    //! Returns builder for SurfaceSet.
    static Builder builder();

 private:
    Array1<std::shared_ptr<Surface<N>>> _surfaces;
    Array1<std::shared_ptr<Surface<N>>> _unboundedSurfaces;
    mutable Bvh<std::shared_ptr<Surface<N>>, N> _bvh;
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

    void invalidateBvh();

    void buildBvh() const;
};

//! 2-D SurfaceSet type.
using SurfaceSet2 = SurfaceSet<2>;

//! 3-D SurfaceSet type.
using SurfaceSet3 = SurfaceSet<3>;

//! Shared pointer for the SurfaceSet2 type.
typedef std::shared_ptr<SurfaceSet2> SurfaceSet2Ptr;

//! Shared pointer for the SurfaceSet3 type.
typedef std::shared_ptr<SurfaceSet3> SurfaceSet3Ptr;

//!
//! \brief Front-end to create SurfaceSet objects step by step.
//!
template <size_t N>
class SurfaceSet<N>::Builder final
    : public SurfaceBuilderBase<N, SurfaceSet<N>> {
 public:
    //! Returns builder with other surfaces.
    Builder& withSurfaces(const ConstArrayView1<std::shared_ptr<Surface<N>>>& others);

    //! Builds SurfaceSet.
    SurfaceSet build() const;

    //! Builds shared pointer of SurfaceSet instance.
    std::shared_ptr<SurfaceSet<N>> makeShared() const;

 private:
    using Base = SurfaceBuilderBase<N, SurfaceSet<N>>;
    using Base::_isNormalFlipped;
    using Base::_transform;

    Array1<std::shared_ptr<Surface<N>>> _surfaces;
};

}  // namespace jet

#endif  // INCLUDE_JET_SURFACE_SET_H_
