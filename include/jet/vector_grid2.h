// Copyright (c) 2016 Doyub Kim

#ifndef INCLUDE_JET_VECTOR_GRID2_H_
#define INCLUDE_JET_VECTOR_GRID2_H_

#include <jet/array_accessor2.h>
#include <jet/grid2.h>
#include <jet/vector_field2.h>
#include <memory>

namespace jet {

class VectorGrid2 : public VectorField2, public Grid2 {
 public:
    typedef ArrayAccessor2<Vector2D> VectorDataAccessor;
    typedef ConstArrayAccessor2<Vector2D> ConstVectorDataAccessor;

    VectorGrid2();

    virtual ~VectorGrid2();

    void clear();

    void resize(
        size_t resolutionX,
        size_t resolutionY,
        double gridSpacingX = 1.0,
        double gridSpacingY = 1.0,
        double originX = 0.0,
        double originY = 0.0,
        double initialValueX = 0.0,
        double initialValueY = 0.0);

    void resize(
        const Size2& resolution,
        const Vector2D& gridSpacing = Vector2D(1, 1),
        const Vector2D& origin = Vector2D(),
        const Vector2D& initialValue = Vector2D());

    void resize(
        double gridSpacingX,
        double gridSpacingY,
        double originX,
        double originY);

    void resize(const Vector2D& gridSpacing, const Vector2D& origin);

    virtual void fill(const Vector2D& value) = 0;

    virtual void fill(const std::function<Vector2D(const Vector2D&)>& func) = 0;

    virtual std::shared_ptr<VectorGrid2> clone() const = 0;

 protected:
    virtual void onResize(
        const Size2& resolution,
        const Vector2D& gridSpacing,
        const Vector2D& origin,
        const Vector2D& initialValue) = 0;
};

typedef std::shared_ptr<VectorGrid2> VectorGrid2Ptr;


class VectorGridBuilder2 {
 public:
    VectorGridBuilder2();

    virtual ~VectorGridBuilder2();

    virtual VectorGrid2Ptr build(
        const Size2& resolution,
        const Vector2D& gridSpacing,
        const Vector2D& gridOrigin,
        const Vector2D& initialVal) const = 0;
};

typedef std::shared_ptr<VectorGridBuilder2> VectorGridBuilder2Ptr;

}  // namespace jet

#endif  // INCLUDE_JET_VECTOR_GRID2_H_
