// Copyright (c) 2016 Doyub Kim

#ifndef INCLUDE_JET_CONSTANT_VECTOR_FIELD3_H_
#define INCLUDE_JET_CONSTANT_VECTOR_FIELD3_H_

#include <jet/vector_field3.h>
#include <memory>

namespace jet {

class ConstantVectorField3 final : public VectorField3 {
 public:
    explicit ConstantVectorField3(const Vector3D& value);

    virtual ~ConstantVectorField3();

    Vector3D sample(const Vector3D& x) const override;

    std::function<Vector3D(const Vector3D&)> sampler() const override;

 private:
    Vector3D _value;
};

typedef std::shared_ptr<ConstantVectorField3> ConstantVectorField3Ptr;

}  // namespace jet

#endif  // INCLUDE_JET_CONSTANT_VECTOR_FIELD3_H_
