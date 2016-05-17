// Copyright (c) 2016 Doyub Kim

#ifndef INCLUDE_JET_GRID_SYSTEM_DATA2_H_
#define INCLUDE_JET_GRID_SYSTEM_DATA2_H_

#include <jet/scalar_grid2.h>
#include <jet/face_centered_grid2.h>
#include <memory>
#include <vector>

namespace jet {

class GridSystemData2 {
 public:
    GridSystemData2();

    virtual ~GridSystemData2();

    void resize(
        const Size2& resolution,
        const Vector2D& gridSpacing,
        const Vector2D& origin);

    Size2 resolution() const;

    Vector2D gridSpacing() const;

    Vector2D origin() const;

    BoundingBox2D boundingBox() const;

    size_t addScalarData(
        const ScalarGridBuilder2Ptr& builder,
        double initialVal = 0.0);

    size_t addVectorData(
        const VectorGridBuilder2Ptr& builder,
        const Vector2D& initialVal = Vector2D());

    size_t addAdvectableScalarData(
        const ScalarGridBuilder2Ptr& builder,
        double initialVal = 0.0);

    size_t addAdvectableVectorData(
        const VectorGridBuilder2Ptr& builder,
        const Vector2D& initialVal = Vector2D());

    const FaceCenteredGrid2Ptr& velocity() const;

    const ScalarGrid2Ptr& scalarDataAt(size_t idx) const;

    const VectorGrid2Ptr& vectorDataAt(size_t idx) const;

    const ScalarGrid2Ptr& advectableScalarDataAt(size_t idx) const;

    const VectorGrid2Ptr& advectableVectorDataAt(size_t idx) const;

    size_t numberOfScalarData() const;

    size_t numberOfVectorData() const;

    size_t numberOfAdvectableScalarData() const;

    size_t numberOfAdvectableVectorData() const;

 private:
    FaceCenteredGrid2Ptr _velocity;
    std::vector<ScalarGrid2Ptr> _scalarDataList;
    std::vector<VectorGrid2Ptr> _vectorDataList;
    std::vector<ScalarGrid2Ptr> _advectableScalarDataList;
    std::vector<VectorGrid2Ptr> _advectableVectorDataList;
};

typedef std::shared_ptr<GridSystemData2> GridSystemData2Ptr;

}  // namespace jet

#endif  // INCLUDE_JET_GRID_SYSTEM_DATA2_H_
