// Copyright (c) 2018 Doyub Kim
//
// I am making my contributions/submissions to this project solely in my
// personal capacity and am not conveying any rights to any intellectual
// property of any third parties.

#ifndef INCLUDE_JET_SPH_POINTS_TO_IMPLICIT3_H_
#define INCLUDE_JET_SPH_POINTS_TO_IMPLICIT3_H_

#include <jet/points_to_implicit3.h>

namespace jet {

//!
//! \brief 3-D points-to-implicit converter based on standard SPH kernel.
//!
//! \see Müller, Matthias, David Charypar, and Markus Gross.
//!      "Particle-based fluid simulation for interactive applications."
//!      Proceedings of the 2003 ACM SIGGRAPH/Eurographics symposium on Computer
//!      animation. Eurographics Association, 2003.
//!
class SphPointsToImplicit3 final : public PointsToImplicit3 {
 public:
    //! Constructs the converter with given kernel radius and cut-off density.
    SphPointsToImplicit3(double kernelRadius = 1.0, double cutOffDensity = 0.5,
                         bool isOutputSdf = true);

    //! Converts the given points to implicit surface scalar field.
    void convert(const ConstArrayAccessor1<Vector3D>& points,
                 ScalarGrid3* output) const override;

 private:
    double _kernelRadius = 1.0;
    double _cutOffDensity = 0.5;
    bool _isOutputSdf = true;
};

//! Shared pointer type for SphPointsToImplicit3 class.
typedef std::shared_ptr<SphPointsToImplicit3> SphPointsToImplicit3Ptr;

}  // namespace jet

#endif  // INCLUDE_JET_SPH_POINTS_TO_IMPLICIT3_H_
