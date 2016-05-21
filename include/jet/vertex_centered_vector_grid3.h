// Copyright (c) 2016 Doyub Kim

#ifndef INCLUDE_JET_VERTEX_CENTERED_VECTOR_GRID3_H_
#define INCLUDE_JET_VERTEX_CENTERED_VECTOR_GRID3_H_

#include <jet/collocated_vector_grid3.h>
#include <algorithm>  // just make cpplint happy..

namespace jet {

class VertexCenteredVectorGrid3 final : public CollocatedVectorGrid3 {
 public:
    VertexCenteredVectorGrid3();

    explicit VertexCenteredVectorGrid3(
        size_t resolutionX,
        size_t resolutionY,
        size_t resolutionZ,
        double gridSpacingX = 1.0,
        double gridSpacingY = 1.0,
        double gridSpacingZ = 1.0,
        double originX = 0.0,
        double originY = 0.0,
        double originZ = 0.0,
        double initialValueU = 0.0,
        double initialValueV = 0.0,
        double initialValueW = 0.0);

    explicit VertexCenteredVectorGrid3(
        const Size3& resolution,
        const Vector3D& gridSpacing = Vector3D(1.0, 1.0, 1.0),
        const Vector3D& origin = Vector3D(),
        const Vector3D& initialValue = Vector3D());

    virtual ~VertexCenteredVectorGrid3();

    Size3 dataSize() const override;

    //! Returns data position for the grid point at (0, 0, 0).
    //! Note that this is different from origin() since origin() returns
    //! the lower corner point of the bounding box.
    Vector3D dataOrigin() const override;

    void swap(Grid3* other) override;

    void fill(const Vector3D& value) override;

    void fill(const std::function<Vector3D(const Vector3D&)>& func) override;

    std::shared_ptr<VectorGrid3> clone() const override;

    static VectorGridBuilder3Ptr builder();
};


class VertexCenteredVectorGridBuilder3 final : public VectorGridBuilder3 {
 public:
    VertexCenteredVectorGridBuilder3();

    VectorGrid3Ptr build(
        const Size3& resolution,
        const Vector3D& gridSpacing,
        const Vector3D& gridOrigin,
        const Vector3D& initialVal) const override;
};

}  // namespace jet

#endif  // INCLUDE_JET_VERTEX_CENTERED_VECTOR_GRID3_H_
