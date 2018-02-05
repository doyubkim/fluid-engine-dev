// Copyright (c) 2018 Doyub Kim
//
// I am making my contributions/submissions to this project solely in my
// personal capacity and am not conveying any rights to any intellectual
// property of any third parties.

#ifndef INCLUDE_JET_CUSTOM_VECTOR_FIELD2_H_
#define INCLUDE_JET_CUSTOM_VECTOR_FIELD2_H_

#include <jet/vector_field2.h>

namespace jet {

//! 2-D vector field with custom field function.
class CustomVectorField2 final : public VectorField2 {
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
    CustomVectorField2(
        const std::function<Vector2D(const Vector2D&)>& customFunction,
        double derivativeResolution = 1e-3);

    //!
    //! \brief Constructs a field with given field and gradient function.
    //!
    //! This constructor creates a field with user-provided field and gradient
    //! function objects. To compute Laplacian, finite differencing is used.
    //! Thus, the differencing resolution also can be provided as the last
    //! parameter.
    //!
    CustomVectorField2(
        const std::function<Vector2D(const Vector2D&)>& customFunction,
        const std::function<double(const Vector2D&)>& customDivergenceFunction,
        double derivativeResolution = 1e-3);

    //! Constructs a field with given field, gradient, and Laplacian function.
    CustomVectorField2(
        const std::function<Vector2D(const Vector2D&)>& customFunction,
        const std::function<double(const Vector2D&)>& customDivergenceFunction,
        const std::function<double(const Vector2D&)>& customCurlFunction);

    //! Returns the sampled value at given position \p x.
    Vector2D sample(const Vector2D& x) const override;

    //! Returns the divergence at given position \p x.
    double divergence(const Vector2D& x) const override;

    //! Returns the curl at given position \p x.
    double curl(const Vector2D& x) const override;

    //! Returns the sampler function.
    std::function<Vector2D(const Vector2D&)> sampler() const override;

    //! Returns builder fox CustomVectorField2.
    static Builder builder();

 private:
    std::function<Vector2D(const Vector2D&)> _customFunction;
    std::function<double(const Vector2D&)> _customDivergenceFunction;
    std::function<double(const Vector2D&)> _customCurlFunction;
    double _resolution = 1e-3;
};

//! Shared pointer type for the CustomVectorField2.
typedef std::shared_ptr<CustomVectorField2> CustomVectorField2Ptr;


//!
//! \brief Front-end to create CustomVectorField2 objects step by step.
//!
class CustomVectorField2::Builder final {
 public:
    //! Returns builder with field function.
    Builder& withFunction(
        const std::function<Vector2D(const Vector2D&)>& func);

    //! Returns builder with divergence function.
    Builder& withDivergenceFunction(
        const std::function<double(const Vector2D&)>& func);

    //! Returns builder with curl function.
    Builder& withCurlFunction(
        const std::function<double(const Vector2D&)>& func);

    //! Returns builder with derivative resolution.
    Builder& withDerivativeResolution(double resolution);

    //! Builds CustomVectorField2.
    CustomVectorField2 build() const;

    //! Builds shared pointer of CustomVectorField2 instance.
    CustomVectorField2Ptr makeShared() const;

 private:
    double _resolution = 1e-3;
    std::function<Vector2D(const Vector2D&)> _customFunction;
    std::function<double(const Vector2D&)> _customDivergenceFunction;
    std::function<double(const Vector2D&)> _customCurlFunction;
};

}  // namespace jet

#endif  // INCLUDE_JET_CUSTOM_VECTOR_FIELD2_H_
