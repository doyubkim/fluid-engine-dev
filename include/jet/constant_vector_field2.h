// Copyright (c) 2018 Doyub Kim
//
// I am making my contributions/submissions to this project solely in my
// personal capacity and am not conveying any rights to any intellectual
// property of any third parties.

#ifndef INCLUDE_JET_CONSTANT_VECTOR_FIELD2_H_
#define INCLUDE_JET_CONSTANT_VECTOR_FIELD2_H_

#include <jet/vector_field2.h>
#include <memory>

namespace jet {

//! 2-D constant vector field.
class ConstantVectorField2 final : public VectorField2 {
 public:
    class Builder;

    //! Constructs a constant vector field with given \p value.
    explicit ConstantVectorField2(const Vector2D& value);

    //! Returns the sampled value at given position \p x.
    Vector2D sample(const Vector2D& x) const override;

    //! Returns the sampler function.
    std::function<Vector2D(const Vector2D&)> sampler() const override;

    //! Returns builder fox ConstantVectorField2.
    static Builder builder();

 private:
    Vector2D _value;
};

//! Shared pointer for the ConstantVectorField2 type.
typedef std::shared_ptr<ConstantVectorField2> ConstantVectorField2Ptr;


//!
//! \brief Front-end to create ConstantVectorField2 objects step by step.
//!
class ConstantVectorField2::Builder final {
 public:
    //! Returns builder with value.
    Builder& withValue(const Vector2D& value);

    //! Builds ConstantVectorField2.
    ConstantVectorField2 build() const;

    //! Builds shared pointer of ConstantVectorField2 instance.
    ConstantVectorField2Ptr makeShared() const;

 private:
    Vector2D _value{0, 0};
};

}  // namespace jet

#endif  // INCLUDE_JET_CONSTANT_VECTOR_FIELD2_H_
