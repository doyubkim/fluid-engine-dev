// Copyright (c) 2018 Doyub Kim
//
// I am making my contributions/submissions to this project solely in my
// personal capacity and am not conveying any rights to any intellectual
// property of any third parties.

#ifndef INCLUDE_JET_GRID_POINT_GENERATOR3_H_
#define INCLUDE_JET_GRID_POINT_GENERATOR3_H_

#include <jet/point_generator3.h>

namespace jet {

//!
//! \brief 3-D regular-grid point generator.
//!
class GridPointGenerator3 final : public PointGenerator3 {
 public:
    //!
    //! \brief Invokes \p callback function for each regular grid points inside
    //! \p boundingBox.
    //!
    //! This function iterates every regular grid points inside \p boundingBox
    //! where \p spacing is the size of the unit cell of regular grid structure.
    //!
    void forEachPoint(
        const BoundingBox3D& boundingBox,
        double spacing,
        const std::function<bool(const Vector3D&)>& callback) const;
};

//! Shared pointer type for the GridPointGenerator3.
typedef std::shared_ptr<GridPointGenerator3> GridPointGenerator3Ptr;

}  // namespace jet

#endif  // INCLUDE_JET_GRID_POINT_GENERATOR3_H_
