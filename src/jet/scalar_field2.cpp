// Copyright (c) 2016 Doyub Kim

#include <pch.h>
#include <jet/scalar_field2.h>

using namespace jet;

ScalarField2::ScalarField2() {
}

ScalarField2::~ScalarField2() {
}

Vector2D ScalarField2::gradient(const Vector2D&) const {
    return Vector2D();
}

double ScalarField2::laplacian(const Vector2D&) const {
    return 0.0;
}

std::function<double(const Vector2D&)> ScalarField2::sampler() const {
    const ScalarField2* self = this;
    return [self](const Vector2D& x) -> double {
        return self->sample(x);
    };
}
