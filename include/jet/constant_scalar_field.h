// Copyright (c) 2018 Doyub Kim
//
// I am making my contributions/submissions to this project solely in my
// personal capacity and am not conveying any rights to any intellectual
// property of any third parties.

#ifndef INCLUDE_JET_CONSTANT_SCALAR_FIELD_H_
#define INCLUDE_JET_CONSTANT_SCALAR_FIELD_H_

#include <jet/scalar_field.h>

namespace jet {

//! N-D constant scalar field.
template <size_t N>
class ConstantScalarField final : public ScalarField<N> {
 public:
    class Builder;

    //! Constructs a constant scalar field with given \p value.
    explicit ConstantScalarField(double value);

    //! Returns the sampled value at given position \p x.
    double sample(const Vector<double, N>& x) const override;

    //! Returns the sampler function.
    std::function<double(const Vector<double, N>&)> sampler() const override;

    //! Returns builder fox ConstantScalarField.
    static Builder builder();

 private:
    double _value = 0.0;
};

//! 2-D ConstantScalarField type.
using ConstantScalarField2 = ConstantScalarField<2>;

//! 3-D ConstantScalarField type.
using ConstantScalarField3 = ConstantScalarField<3>;

//! Shared pointer for the ConstantScalarField2 type.
using ConstantScalarField2Ptr = std::shared_ptr<ConstantScalarField2>;

//! Shared pointer for the ConstantScalarField3 type.
using ConstantScalarField3Ptr = std::shared_ptr<ConstantScalarField3>;

//!
//! \brief Front-end to create ConstantScalarField objects step by step.
//!
template <size_t N>
class ConstantScalarField<N>::Builder final {
 public:
    //! Returns builder with value.
    Builder& withValue(double value);

    //! Builds ConstantScalarField.
    ConstantScalarField build() const;

    //! Builds shared pointer of ConstantScalarField instance.
    std::shared_ptr<ConstantScalarField> makeShared() const;

 private:
    double _value = 0.0;
};

}  // namespace jet

#endif  // INCLUDE_JET_CONSTANT_SCALAR_FIELD_H_
