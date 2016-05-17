// Copyright (c) 2016 Doyub Kim

#ifndef INCLUDE_JET_CONSTANT_VECTOR_FIELD2_H_
#define INCLUDE_JET_CONSTANT_VECTOR_FIELD2_H_

#include <jet/vector_field2.h>
#include <memory>

namespace jet {

class ConstantVectorField2 final : public VectorField2 {
 public:
    explicit ConstantVectorField2(const Vector2D& value);

    virtual ~ConstantVectorField2();

    Vector2D sample(const Vector2D& x) const override;

    std::function<Vector2D(const Vector2D&)> sampler() const override;

 private:
    Vector2D _value;
};

typedef std::shared_ptr<ConstantVectorField2> ConstantVectorField2Ptr;

}  // namespace jet

#endif  // INCLUDE_JET_CONSTANT_VECTOR_FIELD2_H_
