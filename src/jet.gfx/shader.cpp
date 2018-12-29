// Copyright (c) 2018 Doyub Kim
//
// I am making my contributions/submissions to this project solely in my
// personal capacity and am not conveying any rights to any intellectual
// property of any third parties.

#include <common.h>

#include <jet.gfx/shader.h>

namespace jet {
namespace gfx {

Shader::Shader(const RenderParameters &userRenderParams)
    : _userRenderParams(userRenderParams) {
    _defaultRenderParams.add("ModelViewProjection", Matrix4x4F::makeIdentity());
    _defaultRenderParams.add("ViewWidth", 256.f);
    _defaultRenderParams.add("ViewHeight", 256.f);
}

Shader::~Shader() {}

void Shader::bind(const Renderer *renderer) { onBind(renderer); }

void Shader::unbind(const Renderer *renderer) { onUnbind(renderer); }

VertexFormat Shader::vertexFormat() const { return _vertexFormat; }

const RenderParameters &Shader::defaultRenderParameters() const {
    return _defaultRenderParams;
}

const RenderParameters &Shader::userRenderParameters() const {
    return _userRenderParams;
}

void Shader::setModelViewProjectionMatrix(
    const Matrix4x4F &modelViewProjection) {
    _defaultRenderParams.set("ModelViewProjection", modelViewProjection);
}

void Shader::setViewWidth(float viewWidth) {
    _defaultRenderParams.set("ViewWidth", viewWidth);
}

void Shader::setViewHeight(float viewHeight) {
    _defaultRenderParams.set("ViewHeight", viewHeight);
}

void Shader::setUserRenderParameter(const std::string &name, int32_t newValue) {
    _userRenderParams.set(name, newValue);
}

void Shader::setUserRenderParameter(const std::string &name,
                                    uint32_t newValue) {
    _userRenderParams.set(name, newValue);
}

void Shader::setUserRenderParameter(const std::string &name, float newValue) {
    _userRenderParams.set(name, newValue);
}

void Shader::setUserRenderParameter(const std::string &name,
                                    const Vector2F &newValue) {
    _userRenderParams.set(name, newValue);
}

void Shader::setUserRenderParameter(const std::string &name,
                                    const Vector3F &newValue) {
    _userRenderParams.set(name, newValue);
}

void Shader::setUserRenderParameter(const std::string &name,
                                    const Vector4F &newValue) {
    _userRenderParams.set(name, newValue);
}

void Shader::setUserRenderParameter(const std::string &name,
                                    const Matrix4x4F &newValue) {
    _userRenderParams.set(name, newValue);
}

}  // namespace gfx
}  // namespace jet
