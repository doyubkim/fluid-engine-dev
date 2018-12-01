// Copyright (c) 2018 Doyub Kim
//
// I am making my contributions/submissions to this project solely in my
// personal capacity and am not conveying any rights to any intellectual
// property of any third parties.

#include <pch.h>

#include <jet/vector_field.h>

namespace jet {

template <size_t N>
VectorField<N>::VectorField() {}

template <size_t N>
VectorField<N>::~VectorField() {}

template <size_t N>
double VectorField<N>::divergence(const Vector<double, N> &) const {
    return 0.0;
}

template <size_t N>
typename GetCurl<N>::type VectorField<N>::curl(
    const Vector<double, N> &) const {
    return typename GetCurl<N>::type{};
}

template <size_t N>
std::function<Vector<double, N>(const Vector<double, N> &)>
VectorField<N>::sampler() const {
    const VectorField *self = this;
    return [self](const Vector<double, N> &x) -> Vector<double, N> {
        return self->sample(x);
    };
}

template class VectorField<2>;

template class VectorField<3>;

}  // namespace jet
