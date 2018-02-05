// Copyright (c) 2018 Doyub Kim
//
// I am making my contributions/submissions to this project solely in my
// personal capacity and am not conveying any rights to any intellectual
// property of any third parties.

#ifndef INCLUDE_JET_ANISOTROPIC_POINTS_TO_IMPLICIT3_H_
#define INCLUDE_JET_ANISOTROPIC_POINTS_TO_IMPLICIT3_H_

#include <jet/points_to_implicit3.h>

namespace jet {

//!
//! \brief 3-D points-to-implicit converter using Anisotropic kernels.
//!
//! This class converts 3-D points to implicit surface using anisotropic kernels
//! so that the kernels are oriented and stretched to reflect the point
//! distribution more naturally (thus less bumps). The implementation is based
//! on Yu and Turk's 2013 paper with some modifications.
//!
//! \see Yu, Jihun, and Greg Turk. "Reconstructing surfaces of particle-based
//!      fluids using anisotropic kernels." ACM Transactions on Graphics (TOG)
//!      32.1 (2013): 5.
//!
class AnisotropicPointsToImplicit3 final : public PointsToImplicit3 {
 public:
    //!
    //! \brief Constructs the converter with given parameters.
    //!
    //! \param kernelRadius Kernel radius for interpolations.
    //! \param cutOffDensity Iso-contour density value.
    //! \param positionSmoothingFactor Position smoothing factor.
    //! \param minNumNeighbors Minimum number of neighbors to enable anisotropic
    //!                        kernel.
    //!
    AnisotropicPointsToImplicit3(double kernelRadius = 1.0,
                                 double cutOffDensity = 0.5,
                                 double positionSmoothingFactor = 0.5,
                                 size_t minNumNeighbors = 25,
                                 bool isOutputSdf = true);

    //! Converts the given points to implicit surface scalar field.
    void convert(const ConstArrayAccessor1<Vector3D>& points,
                 ScalarGrid3* output) const override;

 private:
    double _kernelRadius = 1.0;
    double _cutOffDensity = 0.5;
    double _positionSmoothingFactor = 0.0;
    size_t _minNumNeighbors = 25;
    bool _isOutputSdf = true;
};

//! Shared pointer for the AnisotropicPointsToImplicit3 type.
typedef std::shared_ptr<AnisotropicPointsToImplicit3>
    AnisotropicPointsToImplicit3Ptr;

}  // namespace jet

#endif  // INCLUDE_JET_ANISOTROPIC_POINTS_TO_IMPLICIT3_H_
