// Copyright (c) 2018 Doyub Kim
//
// I am making my contributions/submissions to this project solely in my
// personal capacity and am not conveying any rights to any intellectual
// property of any third parties.

#include <pch.h>
#include <jet/scalar_field3.h>

using namespace jet;

ScalarField3::ScalarField3() {
}

ScalarField3::~ScalarField3() {
}

Vector3D ScalarField3::gradient(const Vector3D&) const {
    return Vector3D();
}

double ScalarField3::laplacian(const Vector3D&) const {
    return 0.0;
}

std::function<double(const Vector3D&)> ScalarField3::sampler() const {
    const ScalarField3* self = this;
    return [self](const Vector3D& x) -> double {
        return self->sample(x);
    };
}
