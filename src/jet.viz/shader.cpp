// Copyright (c) 2017 Doyub Kim
//
// I am making my contributions/submissions to this project solely in my
// personal capacity and am not conveying any rights to any intellectual
// property of any third parties.

#include <pch.h>

#include <jet.viz/shader.h>

using namespace jet;
using namespace viz;

Shader::Shader(const RenderParameters& userRenderParams)
    : _userRenderParams(userRenderParams) {
    _defaultRenderParams.add("ModelViewProjection", Matrix4x4F::makeIdentity());
    _defaultRenderParams.add("ViewWidth", 800.f);
    _defaultRenderParams.add("ViewHeight", 600.f);
}

Shader::~Shader() {}

void Shader::bind(Renderer* renderer) { onBind(renderer); }

void Shader::unbind(Renderer* renderer) { onUnbind(renderer); }

VertexFormat Shader::vertexFormat() const { return _vertexFormat; }

const RenderParameters& Shader::defaultRenderParameters() const {
    return _defaultRenderParams;
}

const RenderParameters& Shader::userRenderParameters() const {
    return _userRenderParams;
}

void Shader::setModelViewProjectionMatrix(
    const Matrix4x4F& modelViewProjection) {
    _defaultRenderParams.set("ModelViewProjection", modelViewProjection);
}

void Shader::setViewWidth(float viewWidth) {
    _defaultRenderParams.set("ViewWidth", viewWidth);
}

void Shader::setViewHeight(float viewHeight) {
    _defaultRenderParams.set("ViewHeight", viewHeight);
}

void Shader::setUserRenderParameter(const std::string& name,
                                    int32_t newValue) {
    _userRenderParams.set(name, newValue);
}

void Shader::setUserRenderParameter(const std::string& name,
                                    uint32_t newValue) {
    _userRenderParams.set(name, newValue);
}

void Shader::setUserRenderParameter(const std::string& name, float newValue) {
    _userRenderParams.set(name, newValue);
}

void Shader::setUserRenderParameter(const std::string& name,
                                    const Vector2F& newValue) {
    _userRenderParams.set(name, newValue);
}

void Shader::setUserRenderParameter(const std::string& name,
                                    const Vector3F& newValue) {
    _userRenderParams.set(name, newValue);
}

void Shader::setUserRenderParameter(const std::string& name,
                                    const Vector4F& newValue) {
    _userRenderParams.set(name, newValue);
}

void Shader::setUserRenderParameter(const std::string& name,
                                    const Matrix4x4F& newValue) {
    _userRenderParams.set(name, newValue);
}
