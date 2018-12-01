// Copyright (c) 2018 Doyub Kim
//
// I am making my contributions/submissions to this project solely in my
// personal capacity and am not conveying any rights to any intellectual
// property of any third parties.

#ifndef INCLUDE_JET_CUSTOM_VECTOR_FIELD_H_
#define INCLUDE_JET_CUSTOM_VECTOR_FIELD_H_

#include <jet/vector_field.h>

namespace jet {

//! N-D vector field with custom field function.
template <size_t N>
class CustomVectorField final : public VectorField<N> {
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
    CustomVectorField(
        const std::function<Vector<double, N>(const Vector<double, N>&)>&
            customFunction,
        double derivativeResolution = 1e-3);

    //!
    //! \brief Constructs a field with given field and gradient function.
    //!
    //! This constructor creates a field with user-provided field and gradient
    //! function objects. To compute Laplacian, finite differencing is used.
    //! Thus, the differencing resolution also can be provided as the last
    //! parameter.
    //!
    CustomVectorField(
        const std::function<Vector<double, N>(const Vector<double, N>&)>&
            customFunction,
        const std::function<double(const Vector<double, N>&)>&
            customDivergenceFunction,
        double derivativeResolution = 1e-3);

    //! Constructs a field with given field, gradient, and Laplacian function.
    CustomVectorField(
        const std::function<Vector<double, N>(const Vector<double, N>&)>&
            customFunction,
        const std::function<double(const Vector<double, N>&)>&
            customDivergenceFunction,
        const std::function<typename GetCurl<N>::type(
            const Vector<double, N>&)>& customCurlFunction);

    //! Returns the sampled value at given position \p x.
    Vector<double, N> sample(const Vector<double, N>& x) const override;

    //! Returns the divergence at given position \p x.
    double divergence(const Vector<double, N>& x) const override;

    //! Returns the curl at given position \p x.
    typename GetCurl<N>::type curl(const Vector<double, N>& x) const override;

    //! Returns the sampler function.
    std::function<Vector<double, N>(const Vector<double, N>&)> sampler()
        const override;

    //! Returns builder fox CustomVectorField.
    static Builder builder();

 private:
    std::function<Vector<double, N>(const Vector<double, N>&)> _customFunction;
    std::function<double(const Vector<double, N>&)> _customDivergenceFunction;
    std::function<typename GetCurl<N>::type(const Vector<double, N>&)>
        _customCurlFunction;
    double _resolution = 1e-3;
};

//! 2-D CustomVectorField type.
using CustomVectorField2 = CustomVectorField<2>;

//! 3-D CustomVectorField type.
using CustomVectorField3 = CustomVectorField<3>;

//! Shared pointer type for the CustomVectorField2.
using CustomVectorField2Ptr = std::shared_ptr<CustomVectorField2>;

//! Shared pointer type for the CustomVectorField3.
using CustomVectorField3Ptr = std::shared_ptr<CustomVectorField3>;

//!
//! \brief Front-end to create CustomVectorField objects step by step.
//!
template <size_t N>
class CustomVectorField<N>::Builder final {
 public:
    //! Returns builder with field function.
    Builder& withFunction(
        const std::function<Vector<double, N>(const Vector<double, N>&)>& func);

    //! Returns builder with divergence function.
    Builder& withDivergenceFunction(
        const std::function<double(const Vector<double, N>&)>& func);

    //! Returns builder with curl function.
    Builder& withCurlFunction(
        const std::function<typename GetCurl<N>::type(const Vector<double, N>&)>& func);

    //! Returns builder with derivative resolution.
    Builder& withDerivativeResolution(double resolution);

    //! Builds CustomVectorField.
    CustomVectorField build() const;

    //! Builds shared pointer of CustomVectorField instance.
    std::shared_ptr<CustomVectorField> makeShared() const;

 private:
    double _resolution = 1e-3;
    std::function<Vector<double, N>(const Vector<double, N>&)> _customFunction;
    std::function<double(const Vector<double, N>&)> _customDivergenceFunction;
    std::function<typename GetCurl<N>::type(const Vector<double, N>&)>
        _customCurlFunction;
};

}  // namespace jet

#endif  // INCLUDE_JET_CUSTOM_VECTOR_FIELD_H_
