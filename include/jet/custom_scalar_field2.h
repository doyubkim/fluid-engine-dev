// Copyright (c) 2018 Doyub Kim
//
// I am making my contributions/submissions to this project solely in my
// personal capacity and am not conveying any rights to any intellectual
// property of any third parties.

#ifndef INCLUDE_JET_CUSTOM_SCALAR_FIELD2_H_
#define INCLUDE_JET_CUSTOM_SCALAR_FIELD2_H_

#include <jet/scalar_field2.h>

namespace jet {

//! 2-D scalar field with custom field function.
class CustomScalarField2 final : public ScalarField2 {
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
    CustomScalarField2(
        const std::function<double(const Vector2D&)>& customFunction,
        double derivativeResolution = 1e-3);

    //!
    //! \brief Constructs a field with given field and gradient function.
    //!
    //! This constructor creates a field with user-provided field and gradient
    //! function objects. To compute Laplacian, finite differencing is used.
    //! Thus, the differencing resolution also can be provided as the last
    //! parameter.
    //!
    CustomScalarField2(
        const std::function<double(const Vector2D&)>& customFunction,
        const std::function<Vector2D(const Vector2D&)>& customGradientFunction,
        double derivativeResolution = 1e-3);

    //! Constructs a field with given field, gradient, and Laplacian function.
    CustomScalarField2(
        const std::function<double(const Vector2D&)>& customFunction,
        const std::function<Vector2D(const Vector2D&)>& customGradientFunction,
        const std::function<double(const Vector2D&)>& customLaplacianFunction);

    //! Returns the sampled value at given position \p x.
    double sample(const Vector2D& x) const override;

    //! Returns the sampler function.
    std::function<double(const Vector2D&)> sampler() const override;

    //! Returns the gradient vector at given position \p x.
    Vector2D gradient(const Vector2D& x) const override;

    //! Returns the Laplacian at given position \p x.
    double laplacian(const Vector2D& x) const override;

    //! Returns builder fox CustomScalarField2.
    static Builder builder();

 private:
    std::function<double(const Vector2D&)> _customFunction;
    std::function<Vector2D(const Vector2D&)> _customGradientFunction;
    std::function<double(const Vector2D&)> _customLaplacianFunction;
    double _resolution = 1e-3;
};

//! Shared pointer type for the CustomScalarField2.
typedef std::shared_ptr<CustomScalarField2> CustomScalarField2Ptr;


//!
//! \brief Front-end to create CustomScalarField2 objects step by step.
//!
class CustomScalarField2::Builder final {
 public:
    //! Returns builder with field function.
    Builder& withFunction(
        const std::function<double(const Vector2D&)>& func);

    //! Returns builder with divergence function.
    Builder& withGradientFunction(
        const std::function<Vector2D(const Vector2D&)>& func);

    //! Returns builder with curl function.
    Builder& withLaplacianFunction(
        const std::function<double(const Vector2D&)>& func);

    //! Returns builder with derivative resolution.
    Builder& withDerivativeResolution(double resolution);

    //! Builds CustomScalarField2.
    CustomScalarField2 build() const;

    //! Builds shared pointer of CustomScalarField2 instance.
    CustomScalarField2Ptr makeShared() const;

 private:
    double _resolution = 1e-3;
    std::function<double(const Vector2D&)> _customFunction;
    std::function<Vector2D(const Vector2D&)> _customGradientFunction;
    std::function<double(const Vector2D&)> _customLaplacianFunction;
};

}  // namespace jet

#endif  // INCLUDE_JET_CUSTOM_SCALAR_FIELD2_H_
