// Copyright (c) 2018 Doyub Kim
//
// I am making my contributions/submissions to this project solely in my
// personal capacity and am not conveying any rights to any intellectual
// property of any third parties.

#ifndef INCLUDE_JET_CONSTANT_VECTOR_FIELD_H_
#define INCLUDE_JET_CONSTANT_VECTOR_FIELD_H_

#include <jet/vector_field.h>

namespace jet {

//! N-D constant vector field.
template <size_t N>
class ConstantVectorField final : public VectorField<N> {
 public:
    using VectorType = Vector<double, N>;
    using CurlResultType = typename GetCurl<N>::type;

    class Builder;

    //! Constructs a constant vector field with given \p value.
    explicit ConstantVectorField(const Vector<double, N>& value);

    //! Returns the sampled value at given position \p x.
    Vector<double, N> sample(const Vector<double, N>& x) const override;

    //! Returns divergence at given position \p x.
    double divergence(const VectorType& x) const override;

    //! Returns curl at given position \p x.
    CurlResultType curl(const VectorType& x) const override;

    //! Returns the sampler function.
    std::function<Vector<double, N>(const Vector<double, N>&)> sampler()
        const override;

    //! Returns builder fox ConstantVectorField.
    static Builder builder();

 private:
    Vector<double, N> _value;
};

//! 2-D ConstantVectorField type.
using ConstantVectorField2 = ConstantVectorField<2>;

//! 3-D ConstantVectorField type.
using ConstantVectorField3 = ConstantVectorField<3>;

//! Shared pointer for the ConstantVectorField2 type.
using ConstantVectorField2Ptr = std::shared_ptr<ConstantVectorField2>;

//! Shared pointer for the ConstantVectorField3 type.
using ConstantVectorField3Ptr = std::shared_ptr<ConstantVectorField3>;

//!
//! \brief Front-end to create ConstantVectorField objects step by step.
//!
template <size_t N>
class ConstantVectorField<N>::Builder final {
 public:
    //! Returns builder with value.
    Builder& withValue(const Vector<double, N>& value);

    //! Builds ConstantVectorField.
    ConstantVectorField build() const;

    //! Builds shared pointer of ConstantVectorField instance.
    std::shared_ptr<ConstantVectorField> makeShared() const;

 private:
    Vector<double, N> _value;
};

}  // namespace jet

#endif  // INCLUDE_JET_CONSTANT_VECTOR_FIELD_H_
