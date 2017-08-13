// Copyright (c) 2017 Doyub Kim
//
// I am making my contributions/submissions to this project solely in my
// personal capacity and am not conveying any rights to any intellectual
// property of any third parties.

#ifndef INCLUDE_JET_ANISOTROPIC_POINTS_TO_IMPLICIT2_H_
#define INCLUDE_JET_ANISOTROPIC_POINTS_TO_IMPLICIT2_H_

#include <jet/points_to_implicit2.h>

namespace jet {

//!
//! \brief 2-D points-to-implicit converter using Anisotropic kernels.
//!
//! \see Yu, Jihun, and Greg Turk. "Reconstructing surfaces of particle-based
//!      fluids using anisotropic kernels." ACM Transactions on Graphics (TOG)
//!      32.1 (2013): 5.
//!
class AnisotropicPointsToImplicit2 final : public PointsToImplicit2 {
 public:
    //! Constructs the converter with given kernel radius and cut-off density.
    AnisotropicPointsToImplicit2(double kernelRadius = 1.0,
                                 double cutOffDensity = 0.5);

    //! Converts the given points to implicit surface scalar field.
    void convert(const ConstArrayAccessor1<Vector2D>& points,
                 ScalarGrid2* output) const override;

 private:
    double _kernelRadius = 1.0;
    double _cutOffDensity = 0.5;
};

}  // namespace jet

#endif  // INCLUDE_JET_ANISOTROPIC_POINTS_TO_IMPLICIT2_H_
