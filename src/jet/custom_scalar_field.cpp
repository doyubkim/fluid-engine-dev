// Copyright (c) 2018 Doyub Kim
//
// I am making my contributions/submissions to this project solely in my
// personal capacity and am not conveying any rights to any intellectual
// property of any third parties.

#include <common.h>

#include <jet/custom_scalar_field.h>

namespace jet {

template <size_t N>
CustomScalarField<N>::CustomScalarField(
    const std::function<double(const Vector<double, N> &)> &customFunction,
    double derivativeResolution)
    : _customFunction(customFunction), _resolution(derivativeResolution) {}

template <size_t N>
CustomScalarField<N>::CustomScalarField(
    const std::function<double(const Vector<double, N> &)> &customFunction,
    const std::function<Vector<double, N>(const Vector<double, N> &)>
        &customGradientFunction,
    double derivativeResolution)
    : _customFunction(customFunction),
      _customGradientFunction(customGradientFunction),
      _resolution(derivativeResolution) {}

template <size_t N>
CustomScalarField<N>::CustomScalarField(
    const std::function<double(const Vector<double, N> &)> &customFunction,
    const std::function<Vector<double, N>(const Vector<double, N> &)>
        &customGradientFunction,
    const std::function<double(const Vector<double, N> &)>
        &customLaplacianFunction)
    : _customFunction(customFunction),
      _customGradientFunction(customGradientFunction),
      _customLaplacianFunction(customLaplacianFunction),
      _resolution(1e-3) {}

template <size_t N>
double CustomScalarField<N>::sample(const Vector<double, N> &x) const {
    return _customFunction(x);
}

template <size_t N>
std::function<double(const Vector<double, N> &)> CustomScalarField<N>::sampler()
    const {
    return _customFunction;
}

template <size_t N>
Vector<double, N> CustomScalarField<N>::gradient(
    const Vector<double, N> &x) const {
    if (_customGradientFunction) {
        return _customGradientFunction(x);
    } else {
        Vector<double, N> result;
        for (size_t i = 0; i < N; ++i) {
            Vector<double, N> pt;
            pt[i] = 0.5 * _resolution;

            double left = _customFunction(x - pt);
            double right = _customFunction(x + pt);
            result[i] = (right - left) / _resolution;
        }

        return result;
    }
}

template <size_t N>
double CustomScalarField<N>::laplacian(const Vector<double, N> &x) const {
    if (_customLaplacianFunction) {
        return _customLaplacianFunction(x);
    } else {
        double center = _customFunction(x);
        double sum = -4.0 * center;
        for (size_t i = 0; i < N; ++i) {
            Vector<double, N> pt;
            pt[i] = 0.5 * _resolution;

            double left = _customFunction(x - pt);
            double right = _customFunction(x + pt);
            sum += right + left;
        }

        return sum / (_resolution * _resolution);
    }
}

template <size_t N>
typename CustomScalarField<N>::Builder CustomScalarField<N>::builder() {
    return Builder();
}

template <size_t N>
typename CustomScalarField<N>::Builder &
CustomScalarField<N>::Builder::withFunction(
    const std::function<double(const Vector<double, N> &)> &func) {
    _customFunction = func;
    return *this;
}

template <size_t N>
typename CustomScalarField<N>::Builder &
CustomScalarField<N>::Builder::withGradientFunction(
    const std::function<Vector<double, N>(const Vector<double, N> &)> &func) {
    _customGradientFunction = func;
    return *this;
}

template <size_t N>
typename CustomScalarField<N>::Builder &
CustomScalarField<N>::Builder::withLaplacianFunction(
    const std::function<double(const Vector<double, N> &)> &func) {
    _customLaplacianFunction = func;
    return *this;
}

template <size_t N>
typename CustomScalarField<N>::Builder &
CustomScalarField<N>::Builder::withDerivativeResolution(double resolution) {
    _resolution = resolution;
    return *this;
}

template <size_t N>
CustomScalarField<N> CustomScalarField<N>::Builder::build() const {
    if (_customLaplacianFunction) {
        return CustomScalarField(_customFunction, _customGradientFunction,
                                 _customLaplacianFunction);
    } else {
        return CustomScalarField(_customFunction, _customGradientFunction,
                                 _resolution);
    }
}

template <size_t N>
std::shared_ptr<CustomScalarField<N>>
CustomScalarField<N>::Builder::makeShared() const {
    if (_customLaplacianFunction) {
        return std::shared_ptr<CustomScalarField>(
            new CustomScalarField(_customFunction, _customGradientFunction,
                                  _customLaplacianFunction),
            [](CustomScalarField *obj) { delete obj; });
    } else {
        return std::shared_ptr<CustomScalarField>(
            new CustomScalarField(_customFunction, _customGradientFunction,
                                  _resolution),
            [](CustomScalarField *obj) { delete obj; });
    }
}

template class CustomScalarField<2>;

template class CustomScalarField<3>;

}  // namespace jet
