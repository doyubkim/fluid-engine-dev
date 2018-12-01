// Copyright (c) 2018 Doyub Kim
//
// I am making my contributions/submissions to this project solely in my
// personal capacity and am not conveying any rights to any intellectual
// property of any third parties.

#include <pch.h>

#include <jet/scalar_field.h>

namespace jet {

template <size_t N>
ScalarField<N>::ScalarField() {}

template <size_t N>
ScalarField<N>::~ScalarField() {}

template <size_t N>
Vector<double, N> ScalarField<N>::gradient(const Vector<double, N> &) const {
    return Vector<double, N>();
}

template <size_t N>
double ScalarField<N>::laplacian(const Vector<double, N> &) const {
    return 0.0;
}

template <size_t N>
std::function<double(const Vector<double, N> &)> ScalarField<N>::sampler()
    const {
    const ScalarField *self = this;
    return [self](const Vector<double, N> &x) -> double {
        return self->sample(x);
    };
}

template class ScalarField<2>;

template class ScalarField<3>;

}  // namespace jet
