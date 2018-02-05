// Copyright (c) 2018 Doyub Kim
//
// I am making my contributions/submissions to this project solely in my
// personal capacity and am not conveying any rights to any intellectual
// property of any third parties.

#ifndef INCLUDE_JET_SPHERICAL_POINTS_TO_IMPLICIT3_H_
#define INCLUDE_JET_SPHERICAL_POINTS_TO_IMPLICIT3_H_

#include <jet/points_to_implicit3.h>

namespace jet {

//!
//! \brief 3-D points-to-implicit converter based on simple sphere model.
//!
class SphericalPointsToImplicit3 final : public PointsToImplicit3 {
 public:
    //! Constructs the converter with given sphere radius.
    SphericalPointsToImplicit3(double radius = 1.0, bool isOutputSdf = true);

    //! Converts the given points to implicit surface scalar field.
    void convert(const ConstArrayAccessor1<Vector3D>& points,
                 ScalarGrid3* output) const override;

 private:
    double _radius = 1.0;
    bool _isOutputSdf = true;
};

//! Shared pointer type for SphericalPointsToImplicit3.
typedef std::shared_ptr<SphericalPointsToImplicit3>
    SphericalPointsToImplicit3Ptr;

}  // namespace jet

#endif  // INCLUDE_JET_SPHERICAL_POINTS_TO_IMPLICIT3_H_
