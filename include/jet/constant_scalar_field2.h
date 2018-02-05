// Copyright (c) 2018 Doyub Kim
//
// I am making my contributions/submissions to this project solely in my
// personal capacity and am not conveying any rights to any intellectual
// property of any third parties.

#ifndef INCLUDE_JET_CONSTANT_SCALAR_FIELD2_H_
#define INCLUDE_JET_CONSTANT_SCALAR_FIELD2_H_

#include <jet/scalar_field2.h>

namespace jet {

//! 2-D constant scalar field.
class ConstantScalarField2 final : public ScalarField2 {
 public:
    class Builder;

    //! Constructs a constant scalar field with given \p value.
    explicit ConstantScalarField2(double value);

    //! Returns the sampled value at given position \p x.
    double sample(const Vector2D& x) const override;

    //! Returns the sampler function.
    std::function<double(const Vector2D&)> sampler() const override;

    //! Returns builder fox ConstantScalarField2.
    static Builder builder();

 private:
    double _value = 0.0;
};

//! Shared pointer for the ConstantScalarField2 type.
typedef std::shared_ptr<ConstantScalarField2> ConstantScalarField2Ptr;


//!
//! \brief Front-end to create ConstantScalarField2 objects step by step.
//!
class ConstantScalarField2::Builder final {
 public:
    //! Returns builder with value.
    Builder& withValue(double value);

    //! Builds ConstantScalarField2.
    ConstantScalarField2 build() const;

    //! Builds shared pointer of ConstantScalarField2 instance.
    ConstantScalarField2Ptr makeShared() const;

 private:
    double _value = 0.0;
};

}  // namespace jet

#endif  // INCLUDE_JET_CONSTANT_SCALAR_FIELD2_H_
