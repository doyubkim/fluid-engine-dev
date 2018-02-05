// Copyright (c) 2018 Doyub Kim
//
// I am making my contributions/submissions to this project solely in my
// personal capacity and am not conveying any rights to any intellectual
// property of any third parties.

#include <pch.h>
#include <jet/vector_field2.h>

using namespace jet;

VectorField2::VectorField2() {
}

VectorField2::~VectorField2() {
}

double VectorField2::divergence(const Vector2D&) const {
    return 0.0;
}

double VectorField2::curl(const Vector2D&) const {
    return 0.0;
}

std::function<Vector2D(const Vector2D&)> VectorField2::sampler() const {
    const VectorField2* self = this;
    return [self](const Vector2D& x) -> Vector2D {
        return self->sample(x);
    };
}
