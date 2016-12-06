// Copyright (c) 2016 Doyub Kim

#ifndef INCLUDE_JET_CONSTANT_VECTOR_FIELD3_H_
#define INCLUDE_JET_CONSTANT_VECTOR_FIELD3_H_

#include <jet/vector_field3.h>
#include <memory>

namespace jet {

//! 3-D constant vector field.
class ConstantVectorField3 final : public VectorField3 {
 public:
    //! Constructs a constant vector field with given \p value.
    explicit ConstantVectorField3(const Vector3D& value);

    //! Returns the sampled value at given position \p x.
    Vector3D sample(const Vector3D& x) const override;

    //! Returns the sampler function.
    std::function<Vector3D(const Vector3D&)> sampler() const override;

 private:
    Vector3D _value;
};

//! Shared pointer for the ConstantVectorField3 type.
typedef std::shared_ptr<ConstantVectorField3> ConstantVectorField3Ptr;

}  // namespace jet

#endif  // INCLUDE_JET_CONSTANT_VECTOR_FIELD3_H_
