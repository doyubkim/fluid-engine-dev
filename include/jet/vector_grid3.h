// Copyright (c) 2016 Doyub Kim

#ifndef INCLUDE_JET_VECTOR_GRID3_H_
#define INCLUDE_JET_VECTOR_GRID3_H_

#include <jet/array_accessor3.h>
#include <jet/grid3.h>
#include <jet/vector_field3.h>
#include <memory>

namespace jet {

class VectorGrid3 : public VectorField3, public Grid3 {
 public:
    typedef ArrayAccessor3<Vector3D> VectorDataAccessor;
    typedef ConstArrayAccessor3<Vector3D> ConstVectorDataAccessor;

    VectorGrid3();

    virtual ~VectorGrid3();

    void clear();

    void resize(
        size_t resolutionX,
        size_t resolutionY,
        size_t resolutionZ,
        double gridSpacingX = 1.0,
        double gridSpacingY = 1.0,
        double gridSpacingZ = 1.0,
        double originX = 0.0,
        double originY = 0.0,
        double originZ = 0.0,
        double initialValueX = 0.0,
        double initialValueY = 0.0,
        double initialValueZ = 0.0);

    void resize(
        const Size3& resolution,
        const Vector3D& gridSpacing = Vector3D(1, 1, 1),
        const Vector3D& origin = Vector3D(),
        const Vector3D& initialValue = Vector3D());

    void resize(
        double gridSpacingX,
        double gridSpacingY,
        double gridSpacingZ,
        double originX,
        double originY,
        double originZ);

    void resize(const Vector3D& gridSpacing, const Vector3D& origin);

    virtual void fill(const Vector3D& value) = 0;

    virtual void fill(const std::function<Vector3D(const Vector3D&)>& func) = 0;

    virtual std::shared_ptr<VectorGrid3> clone() const = 0;

 protected:
    virtual void onResize(
        const Size3& resolution,
        const Vector3D& gridSpacing,
        const Vector3D& origin,
        const Vector3D& initialValue) = 0;
};

typedef std::shared_ptr<VectorGrid3> VectorGrid3Ptr;


class VectorGridBuilder3 {
 public:
    VectorGridBuilder3();

    virtual ~VectorGridBuilder3();

    virtual VectorGrid3Ptr build(
        const Size3& resolution,
        const Vector3D& gridSpacing,
        const Vector3D& gridOrigin,
        const Vector3D& initialVal) const = 0;
};

typedef std::shared_ptr<VectorGridBuilder3> VectorGridBuilder3Ptr;

}  // namespace jet

#endif  // INCLUDE_JET_VECTOR_GRID3_H_
