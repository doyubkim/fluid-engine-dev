// Copyright (c) 2018 Doyub Kim
//
// I am making my contributions/submissions to this project solely in my
// personal capacity and am not conveying any rights to any intellectual
// property of any third parties.

#ifndef INCLUDE_JET_CUSTOM_SCALAR_FIELD_H_
#define INCLUDE_JET_CUSTOM_SCALAR_FIELD_H_

#include <jet/scalar_field.h>

namespace jet {

//! N-D scalar field with custom field function.
template <size_t N>
class CustomScalarField final : public ScalarField<N> {
 public:
    class Builder;

    //!
    //! \brief Constructs a field with given function.
    //!
    //! This constructor creates a field with user-provided function object.
    //! To compute derivatives, such as gradient and Laplacian, finite
    //! differencing is used. Thus, the differencing resolution also can be
    //! provided as the last parameter.
    //!
    CustomScalarField(
        const std::function<double(const Vector<double, N>&)>& customFunction,
        double derivativeResolution = 1e-3);

    //!
    //! \brief Constructs a field with given field and gradient function.
    //!
    //! This constructor creates a field with user-provided field and gradient
    //! function objects. To compute Laplacian, finite differencing is used.
    //! Thus, the differencing resolution also can be provided as the last
    //! parameter.
    //!
    CustomScalarField(
        const std::function<double(const Vector<double, N>&)>& customFunction,
        const std::function<Vector<double, N>(const Vector<double, N>&)>&
            customGradientFunction,
        double derivativeResolution = 1e-3);

    //! Constructs a field with given field, gradient, and Laplacian function.
    CustomScalarField(
        const std::function<double(const Vector<double, N>&)>& customFunction,
        const std::function<Vector<double, N>(const Vector<double, N>&)>&
            customGradientFunction,
        const std::function<double(const Vector<double, N>&)>&
            customLaplacianFunction);

    //! Returns the sampled value at given position \p x.
    double sample(const Vector<double, N>& x) const override;

    //! Returns the sampler function.
    std::function<double(const Vector<double, N>&)> sampler() const override;

    //! Returns the gradient vector at given position \p x.
    Vector<double, N> gradient(const Vector<double, N>& x) const override;

    //! Returns the Laplacian at given position \p x.
    double laplacian(const Vector<double, N>& x) const override;

    //! Returns builder fox CustomScalarField.
    static Builder builder();

 private:
    std::function<double(const Vector<double, N>&)> _customFunction;
    std::function<Vector<double, N>(const Vector<double, N>&)>
        _customGradientFunction;
    std::function<double(const Vector<double, N>&)> _customLaplacianFunction;
    double _resolution = 1e-3;
};

//! 2-D CustomScalarField type.
using CustomScalarField2 = CustomScalarField<2>;

//! 3-D CustomScalarField type.
using CustomScalarField3 = CustomScalarField<3>;

//! Shared pointer type for the CustomScalarField2.
using CustomScalarField2Ptr = std::shared_ptr<CustomScalarField2>;

//! Shared pointer type for the CustomScalarField3.
using CustomScalarField3Ptr = std::shared_ptr<CustomScalarField3>;

//!
//! \brief Front-end to create CustomScalarField objects step by step.
//!
template <size_t N>
class CustomScalarField<N>::Builder final {
 public:
    //! Returns builder with field function.
    Builder& withFunction(
        const std::function<double(const Vector<double, N>&)>& func);

    //! Returns builder with divergence function.
    Builder& withGradientFunction(
        const std::function<Vector<double, N>(const Vector<double, N>&)>& func);

    //! Returns builder with curl function.
    Builder& withLaplacianFunction(
        const std::function<double(const Vector<double, N>&)>& func);

    //! Returns builder with derivative resolution.
    Builder& withDerivativeResolution(double resolution);

    //! Builds CustomScalarField.
    CustomScalarField<N> build() const;

    //! Builds shared pointer of CustomScalarField instance.
    std::shared_ptr<CustomScalarField<N>> makeShared() const;

 private:
    double _resolution = 1e-3;
    std::function<double(const Vector<double, N>&)> _customFunction;
    std::function<Vector<double, N>(const Vector<double, N>&)>
        _customGradientFunction;
    std::function<double(const Vector<double, N>&)> _customLaplacianFunction;
};

}  // namespace jet

#endif  // INCLUDE_JET_CUSTOM_SCALAR_FIELD_H_
