// Copyright (c) 2018 Doyub Kim
//
// I am making my contributions/submissions to this project solely in my
// personal capacity and am not conveying any rights to any intellectual
// property of any third parties.

#ifndef INCLUDE_JET_GRID_POINT_GENERATOR2_H_
#define INCLUDE_JET_GRID_POINT_GENERATOR2_H_

#include <jet/point_generator2.h>

namespace jet {

//!
//! \brief 2-D regular-grid point generator.
//!
class GridPointGenerator2 final : public PointGenerator2 {
 public:
    //!
    //! \brief Invokes \p callback function for each regular grid points inside
    //! \p boundingBox.
    //!
    //! This function iterates every regular grid points inside \p boundingBox
    //! where \p spacing is the size of the unit cell of regular grid structure.
    //!
    void forEachPoint(
        const BoundingBox2D& boundingBox,
        double spacing,
        const std::function<bool(const Vector2D&)>& callback) const;
};

//! Shared pointer type for the GridPointGenerator2.
typedef std::shared_ptr<GridPointGenerator2> GridPointGenerator2Ptr;

}  // namespace jet

#endif  // INCLUDE_JET_GRID_POINT_GENERATOR2_H_
