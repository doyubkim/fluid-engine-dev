// Copyright (c) 2018 Doyub Kim
//
// I am making my contributions/submissions to this project solely in my
// personal capacity and am not conveying any rights to any intellectual
// property of any third parties.

#include <common.h>

#include <jet.gfx/renderer.h>

namespace jet {

namespace gfx {

Renderer::Renderer() {}

Renderer::~Renderer() {}

const CameraPtr& Renderer::camera() const { return _camera; }

void Renderer::setCamera(const CameraPtr& camera) { _camera = camera; }

const Vector4D& Renderer::backgroundColor() const { return _bgColor; }

void Renderer::setBackgroundColor(const Vector4D& color) { _bgColor = color; }

}  // namespace gfx

}  // namespace jet
