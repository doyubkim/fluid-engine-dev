// Copyright (c) 2018 Doyub Kim
//
// I am making my contributions/submissions to this project solely in my
// personal capacity and am not conveying any rights to any intellectual
// property of any third parties.

#ifndef INCLUDE_JET_IMPLICIT_SURFACE_H_
#define INCLUDE_JET_IMPLICIT_SURFACE_H_

#include <jet/surface.h>

namespace jet {

//! Abstract base class for N-D implicit surface.
template <size_t N>
class ImplicitSurface : public Surface<N> {
 public:
    using Surface<N>::transform;
    using Surface<N>::isNormalFlipped;

    //! Default constructor.
    ImplicitSurface(const Transform<N>& transform = Transform<N>(),
                    bool isNormalFlipped = false);

    //! Copy constructor.
    ImplicitSurface(const ImplicitSurface& other);

    //! Default destructor.
    virtual ~ImplicitSurface();

    //! Returns signed distance from the given point \p otherPoint.
    double signedDistance(const Vector<double, N>& otherPoint) const;

 protected:
    //! Returns signed distance from the given point \p otherPoint in local
    //! space.
    virtual double signedDistanceLocal(
        const Vector<double, N>& otherPoint) const = 0;

 private:
    double closestDistanceLocal(
        const Vector<double, N>& otherPoint) const override;
};

//! 2-D ImplicitSurface type.
using ImplicitSurface2 = ImplicitSurface<2>;

//! 3-D ImplicitSurface type.
using ImplicitSurface3 = ImplicitSurface<3>;

//! Shared pointer type for the ImplicitSurface2.
using ImplicitSurface2Ptr = std::shared_ptr<ImplicitSurface2>;

//! Shared pointer type for the ImplicitSurface3.
using ImplicitSurface3Ptr = std::shared_ptr<ImplicitSurface3>;

}  // namespace jet

#endif  // INCLUDE_JET_IMPLICIT_SURFACE_H_
