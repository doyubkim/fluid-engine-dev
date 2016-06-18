// Copyright (c) 2016 Doyub Kim

#ifndef INCLUDE_JET_CUSTOM_SCALAR_FIELD2_H_
#define INCLUDE_JET_CUSTOM_SCALAR_FIELD2_H_

#include <jet/scalar_field2.h>

namespace jet {

//! 2-D scalar field with custom field function.
class CustomScalarField2 final : public ScalarField2 {
 public:
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

 private:
    std::function<double(const Vector2D&)> _customFunction;
    std::function<Vector2D(const Vector2D&)> _customGradientFunction;
    std::function<double(const Vector2D&)> _customLaplacianFunction;
    double _resolution = 1e-3;
};

}  // namespace jet

#endif  // INCLUDE_JET_CUSTOM_SCALAR_FIELD2_H_
