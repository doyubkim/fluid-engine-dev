// Copyright (c) 2018 Doyub Kim
//
// I am making my contributions/submissions to this project solely in my
// personal capacity and am not conveying any rights to any intellectual
// property of any third parties.

#include <pch.h>

#include <jet/constant_vector_field.h>

namespace jet {

template <size_t N>
ConstantVectorField<N>::ConstantVectorField(const Vector<double, N> &value)
    : _value(value) {}

template <size_t N>
Vector<double, N> ConstantVectorField<N>::sample(
    const Vector<double, N> &x) const {
    UNUSED_VARIABLE(x);

    return _value;
}

template <size_t N>
std::function<Vector<double, N>(const Vector<double, N> &)>
ConstantVectorField<N>::sampler() const {
    return [this](const Vector<double, N> &) -> Vector<double, N> {
        return _value;
    };
}

template <size_t N>
typename ConstantVectorField<N>::Builder ConstantVectorField<N>::builder() {
    return Builder();
}

template <size_t N>
typename ConstantVectorField<N>::Builder &
ConstantVectorField<N>::Builder::withValue(const Vector<double, N> &value) {
    _value = value;
    return *this;
}

template <size_t N>
ConstantVectorField<N> ConstantVectorField<N>::Builder::build() const {
    return ConstantVectorField(_value);
}

template <size_t N>
std::shared_ptr<ConstantVectorField<N>>
ConstantVectorField<N>::Builder::makeShared() const {
    return std::shared_ptr<ConstantVectorField>(
        new ConstantVectorField(_value),
        [](ConstantVectorField *obj) { delete obj; });
}

template class ConstantVectorField<2>;

template class ConstantVectorField<3>;

}  // namespace jet
