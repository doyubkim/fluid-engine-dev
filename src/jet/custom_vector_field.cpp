// Copyright (c) 2018 Doyub Kim
//
// I am making my contributions/submissions to this project solely in my
// personal capacity and am not conveying any rights to any intellectual
// property of any third parties.

#include <pch.h>

#include <jet/custom_vector_field.h>

namespace jet {

namespace internal {

double curl(std::function<Vector2D(const Vector2D &)> func, const Vector2D &x,
            double resolution) {
    Vector2D left = func(x - Vector2D(0.5 * resolution, 0.0));
    Vector2D right = func(x + Vector2D(0.5 * resolution, 0.0));
    Vector2D bottom = func(x - Vector2D(0.0, 0.5 * resolution));
    Vector2D top = func(x + Vector2D(0.0, 0.5 * resolution));

    double Fx_ym = bottom.x;
    double Fx_yp = top.x;

    double Fy_xm = left.y;
    double Fy_xp = right.y;

    return 0.5 * (Fy_xp - Fy_xm) / resolution -
           0.5 * (Fx_yp - Fx_ym) / resolution;
}

Vector3D curl(std::function<Vector3D(const Vector3D &)> func, const Vector3D &x,
              double resolution) {
    Vector3D left = func(x - Vector3D(0.5 * resolution, 0.0, 0.0));
    Vector3D right = func(x + Vector3D(0.5 * resolution, 0.0, 0.0));
    Vector3D bottom = func(x - Vector3D(0.0, 0.5 * resolution, 0.0));
    Vector3D top = func(x + Vector3D(0.0, 0.5 * resolution, 0.0));
    Vector3D back = func(x - Vector3D(0.0, 0.0, 0.5 * resolution));
    Vector3D front = func(x + Vector3D(0.0, 0.0, 0.5 * resolution));

    double Fx_ym = bottom.x;
    double Fx_yp = top.x;
    double Fx_zm = back.x;
    double Fx_zp = front.x;

    double Fy_xm = left.y;
    double Fy_xp = right.y;
    double Fy_zm = back.y;
    double Fy_zp = front.y;

    double Fz_xm = left.z;
    double Fz_xp = right.z;
    double Fz_ym = bottom.z;
    double Fz_yp = top.z;

    return Vector3D(
        0.5 * (Fz_yp - Fz_ym) / resolution - 0.5 * (Fy_zp - Fy_zm) / resolution,
        0.5 * (Fx_zp - Fx_zm) / resolution - 0.5 * (Fz_xp - Fz_xm) / resolution,
        0.5 * (Fy_xp - Fy_xm) / resolution -
            0.5 * (Fx_yp - Fx_ym) / resolution);
}
}  // namespace internal

template <size_t N>
CustomVectorField<N>::CustomVectorField(
    const std::function<Vector<double, N>(const Vector<double, N> &)>
        &customFunction,
    double derivativeResolution)
    : _customFunction(customFunction), _resolution(derivativeResolution) {}

template <size_t N>
CustomVectorField<N>::CustomVectorField(
    const std::function<Vector<double, N>(const Vector<double, N> &)>
        &customFunction,
    const std::function<double(const Vector<double, N> &)>
        &customDivergenceFunction,
    double derivativeResolution)
    : _customFunction(customFunction),
      _customDivergenceFunction(customDivergenceFunction),
      _resolution(derivativeResolution) {}

template <size_t N>
CustomVectorField<N>::CustomVectorField(
    const std::function<Vector<double, N>(const Vector<double, N> &)>
        &customFunction,
    const std::function<double(const Vector<double, N> &)>
        &customDivergenceFunction,
    const std::function<typename GetCurl<N>::type(const Vector<double, N> &)>
        &customCurlFunction)
    : _customFunction(customFunction),
      _customDivergenceFunction(customDivergenceFunction),
      _customCurlFunction(customCurlFunction) {}

template <size_t N>
Vector<double, N> CustomVectorField<N>::sample(
    const Vector<double, N> &x) const {
    return _customFunction(x);
}

template <size_t N>
std::function<Vector<double, N>(const Vector<double, N> &)>
CustomVectorField<N>::sampler() const {
    return _customFunction;
}

template <size_t N>
double CustomVectorField<N>::divergence(const Vector<double, N> &x) const {
    if (_customDivergenceFunction) {
        return _customDivergenceFunction(x);
    } else {
        double sum = 0.0;
        for (size_t i = 0; i < N; ++i) {
            Vector<double, N> pt;
            pt[i] = 0.5 * _resolution;

            double left = _customFunction(x - pt)[i];
            double right = _customFunction(x + pt)[i];
            sum += right - left;
        }

        return sum / _resolution;
    }
}

template <size_t N>
typename GetCurl<N>::type CustomVectorField<N>::curl(
    const Vector<double, N> &x) const {
    if (_customCurlFunction) {
        return _customCurlFunction(x);
    } else {
        return internal::curl(_customFunction, x, _resolution);
    }
}

template <size_t N>
typename CustomVectorField<N>::Builder CustomVectorField<N>::builder() {
    return Builder();
}

template <size_t N>
typename CustomVectorField<N>::Builder &
CustomVectorField<N>::Builder::withFunction(
    const std::function<Vector<double, N>(const Vector<double, N> &)> &func) {
    _customFunction = func;
    return *this;
}

template <size_t N>
typename CustomVectorField<N>::Builder &
CustomVectorField<N>::Builder::withDivergenceFunction(
    const std::function<double(const Vector<double, N> &)> &func) {
    _customDivergenceFunction = func;
    return *this;
}

template <size_t N>
typename CustomVectorField<N>::Builder &
CustomVectorField<N>::Builder::withCurlFunction(
    const std::function<typename GetCurl<N>::type(const Vector<double, N> &)>
        &func) {
    _customCurlFunction = func;
    return *this;
}

template <size_t N>
typename CustomVectorField<N>::Builder &
CustomVectorField<N>::Builder::withDerivativeResolution(double resolution) {
    _resolution = resolution;
    return *this;
}

template <size_t N>
CustomVectorField<N> CustomVectorField<N>::Builder::build() const {
    if (_customCurlFunction) {
        return CustomVectorField(_customFunction, _customDivergenceFunction,
                                 _customCurlFunction);
    } else {
        return CustomVectorField(_customFunction, _customDivergenceFunction,
                                 _resolution);
    }
}

template <size_t N>
std::shared_ptr<CustomVectorField<N>>
CustomVectorField<N>::Builder::makeShared() const {
    if (_customCurlFunction) {
        return std::shared_ptr<CustomVectorField>(
            new CustomVectorField(_customFunction, _customDivergenceFunction,
                                  _customCurlFunction),
            [](CustomVectorField *obj) { delete obj; });
    } else {
        return std::shared_ptr<CustomVectorField>(
            new CustomVectorField(_customFunction, _customDivergenceFunction,
                                  _resolution),
            [](CustomVectorField *obj) { delete obj; });
    }
}

template class CustomVectorField<2>;

template class CustomVectorField<3>;

}  // namespace jet
