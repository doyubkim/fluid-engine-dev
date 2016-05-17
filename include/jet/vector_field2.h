// Copyright (c) 2016 Doyub Kim

#ifndef INCLUDE_JET_VECTOR_FIELD2_H_
#define INCLUDE_JET_VECTOR_FIELD2_H_

#include <jet/field2.h>
#include <jet/vector2.h>
#include <functional>
#include <memory>

namespace jet {

class VectorField2 : public Field2 {
 public:
    VectorField2();

    virtual ~VectorField2();

    virtual Vector2D sample(const Vector2D& x) const = 0;

    virtual double divergence(const Vector2D& x) const;

    virtual double curl(const Vector2D& x) const;

    virtual std::function<Vector2D(const Vector2D&)> sampler() const;
};

typedef std::shared_ptr<VectorField2> VectorField2Ptr;

}  // namespace jet

#endif  // INCLUDE_JET_VECTOR_FIELD2_H_
