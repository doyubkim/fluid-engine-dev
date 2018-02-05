// Copyright (c) 2018 Doyub Kim
//
// I am making my contributions/submissions to this project solely in my
// personal capacity and am not conveying any rights to any intellectual
// property of any third parties.

#ifndef INCLUDE_JET_TRIANGLE_POINT_GENERATOR_H_
#define INCLUDE_JET_TRIANGLE_POINT_GENERATOR_H_

#include <jet/point_generator2.h>

namespace jet {

//!
//! \brief Right triangle point generator.
//!
class TrianglePointGenerator final : public PointGenerator2 {
 public:
    //!
    //! \brief Invokes \p callback function for each right triangle points
    //! inside \p boundingBox.
    //!
    //! This function iterates every right triangle points inside \p boundingBox
    //! where \p spacing is the size of the right triangle structure.
    //!
    void forEachPoint(
        const BoundingBox2D& boundingBox,
        double spacing,
        const std::function<bool(const Vector2D&)>& callback) const override;
};

typedef std::shared_ptr<TrianglePointGenerator> TrianglePointGeneratorPtr;

}  // namespace jet

#endif  // INCLUDE_JET_TRIANGLE_POINT_GENERATOR_H_
