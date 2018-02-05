// Copyright (c) 2018 Doyub Kim
//
// I am making my contributions/submissions to this project solely in my
// personal capacity and am not conveying any rights to any intellectual
// property of any third parties.

#ifndef INCLUDE_JET_ZHU_BRIDSON_POINTS_TO_IMPLICIT2_H_
#define INCLUDE_JET_ZHU_BRIDSON_POINTS_TO_IMPLICIT2_H_

#include <jet/points_to_implicit2.h>

namespace jet {

//!
//! \brief 2-D points-to-implicit converter based on Zhu and Bridson's method.
//!
//! \see Zhu, Yongning, and Robert Bridson. "Animating sand as a fluid."
//!      ACM Transactions on Graphics (TOG). Vol. 24. No. 3. ACM, 2005.
//!
class ZhuBridsonPointsToImplicit2 final : public PointsToImplicit2 {
 public:
    //! Constructs the converter with given kernel radius and cut-off threshold.
    ZhuBridsonPointsToImplicit2(double kernelRadius = 1.0,
                                double cutOffThreshold = 0.25,
                                bool isOutputSdf = true);

    //! Converts the given points to implicit surface scalar field.
    void convert(const ConstArrayAccessor1<Vector2D>& points,
                 ScalarGrid2* output) const override;

 private:
    double _kernelRadius = 1.0;
    double _cutOffThreshold = 0.25;
    bool _isOutputSdf = true;
};

//! Shared pointer type for ZhuBridsonPointsToImplicit2 class
typedef std::shared_ptr<ZhuBridsonPointsToImplicit2> ZhuBridsonPointsToImplicit2Ptr;

}  // namespace jet

#endif  // INCLUDE_JET_ZHU_BRIDSON_POINTS_TO_IMPLICIT2_H_
