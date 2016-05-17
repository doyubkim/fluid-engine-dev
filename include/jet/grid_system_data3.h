// Copyright (c) 2016 Doyub Kim

#ifndef INCLUDE_JET_GRID_SYSTEM_DATA3_H_
#define INCLUDE_JET_GRID_SYSTEM_DATA3_H_

#include <jet/scalar_grid3.h>
#include <jet/face_centered_grid3.h>
#include <memory>
#include <vector>

namespace jet {

class GridSystemData3 {
 public:
    GridSystemData3();

    virtual ~GridSystemData3();

    void resize(
        const Size3& resolution,
        const Vector3D& gridSpacing,
        const Vector3D& origin);

    Size3 resolution() const;

    Vector3D gridSpacing() const;

    Vector3D origin() const;

    BoundingBox3D boundingBox() const;

    size_t addScalarData(
        const ScalarGridBuilder3Ptr& builder,
        double initialVal = 0.0);

    size_t addVectorData(
        const VectorGridBuilder3Ptr& builder,
        const Vector3D& initialVal = Vector3D());

    size_t addAdvectableScalarData(
        const ScalarGridBuilder3Ptr& builder,
        double initialVal = 0.0);

    size_t addAdvectableVectorData(
        const VectorGridBuilder3Ptr& builder,
        const Vector3D& initialVal = Vector3D());

    const FaceCenteredGrid3Ptr& velocity() const;

    const ScalarGrid3Ptr& scalarDataAt(size_t idx) const;

    const VectorGrid3Ptr& vectorDataAt(size_t idx) const;

    const ScalarGrid3Ptr& advectableScalarDataAt(size_t idx) const;

    const VectorGrid3Ptr& advectableVectorDataAt(size_t idx) const;

    size_t numberOfScalarData() const;

    size_t numberOfVectorData() const;

    size_t numberOfAdvectableScalarData() const;

    size_t numberOfAdvectableVectorData() const;

 private:
    FaceCenteredGrid3Ptr _velocity;
    std::vector<ScalarGrid3Ptr> _scalarDataList;
    std::vector<VectorGrid3Ptr> _vectorDataList;
    std::vector<ScalarGrid3Ptr> _advectableScalarDataList;
    std::vector<VectorGrid3Ptr> _advectableVectorDataList;
};

typedef std::shared_ptr<GridSystemData3> GridSystemData3Ptr;

}  // namespace jet

#endif  // INCLUDE_JET_GRID_SYSTEM_DATA3_H_
