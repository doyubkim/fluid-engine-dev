// Copyright (c) 2016 Doyub Kim

#ifndef INCLUDE_JET_VERTEX_CENTERED_SCALAR_GRID3_H_
#define INCLUDE_JET_VERTEX_CENTERED_SCALAR_GRID3_H_

#include <jet/array3.h>
#include <jet/scalar_grid3.h>
#include <algorithm>  // just make cpplint happy..

namespace jet {

class VertexCenteredScalarGrid3 final : public ScalarGrid3 {
 public:
    VertexCenteredScalarGrid3();

    explicit VertexCenteredScalarGrid3(
         size_t resolutionX,
         size_t resolutionY,
         size_t resolutionZ,
         double gridSpacingX = 1.0,
         double gridSpacingY = 1.0,
         double gridSpacingZ = 1.0,
         double originX = 0.0,
         double originY = 0.0,
         double originZ = 0.0,
         double initialValue = 0.0);

    explicit VertexCenteredScalarGrid3(
        const Size3& resolution,
        const Vector3D& gridSpacing = Vector3D(1.0, 1.0, 1.0),
        const Vector3D& origin = Vector3D(),
        double initialValue = 0.0);

    virtual ~VertexCenteredScalarGrid3();

    Size3 dataSize() const override;

    //! Returns data position for the grid point at (0, 0, 0).
    //! Note that this is different from origin() since origin() returns
    //! the lower corner point of the bounding box.
    Vector3D dataOrigin() const override;

    std::shared_ptr<ScalarGrid3> clone() const override;

    void swap(Grid3* other) override;

    static ScalarGridBuilder3Ptr builder();
};


class VertexCenteredScalarGridBuilder3 final : public ScalarGridBuilder3 {
 public:
    VertexCenteredScalarGridBuilder3();

    ScalarGrid3Ptr build(
        const Size3& resolution,
        const Vector3D& gridSpacing,
        const Vector3D& gridOrigin,
        double initialVal) const override;
};

}  // namespace jet

#endif  // INCLUDE_JET_VERTEX_CENTERED_SCALAR_GRID3_H_
