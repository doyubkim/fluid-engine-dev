// Copyright (c) 2018 Doyub Kim
//
// I am making my contributions/submissions to this project solely in my
// personal capacity and am not conveying any rights to any intellectual
// property of any third parties.

#include <common.h>

#include <jet/constant_scalar_field.h>

namespace jet {

template <size_t N>
ConstantScalarField<N>::ConstantScalarField(double value) : _value(value) {}

template <size_t N>
double ConstantScalarField<N>::sample(const Vector<double, N> &x) const {
    UNUSED_VARIABLE(x);

    return _value;
}

template <size_t N>
std::function<double(const Vector<double, N> &)>
ConstantScalarField<N>::sampler() const {
    double value = _value;
    return [value](const Vector<double, N> &) -> double { return value; };
}

template <size_t N>
typename ConstantScalarField<N>::Builder ConstantScalarField<N>::builder() {
    return Builder();
}

template <size_t N>
typename ConstantScalarField<N>::Builder &
ConstantScalarField<N>::Builder::withValue(double value) {
    _value = value;
    return *this;
}

template <size_t N>
ConstantScalarField<N> ConstantScalarField<N>::Builder::build() const {
    return ConstantScalarField(_value);
}

template <size_t N>
std::shared_ptr<ConstantScalarField<N>>
ConstantScalarField<N>::Builder::makeShared() const {
    return std::shared_ptr<ConstantScalarField>(
        new ConstantScalarField(_value),
        [](ConstantScalarField *obj) { delete obj; });
}

template class ConstantScalarField<2>;

template class ConstantScalarField<3>;

}  // namespace jet
