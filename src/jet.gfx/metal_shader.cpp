// Copyright (c) 2018 Doyub Kim
//
// I am making my contributions/submissions to this project solely in my
// personal capacity and am not conveying any rights to any intellectual
// property of any third parties.

#include <common.h>

#include "mtlpp_wrappers.h"

#include <jet.gfx/metal_renderer.h>
#include <jet.gfx/metal_shader.h>

namespace jet {
namespace gfx {

MetalShader::MetalShader(const std::string &name,
                         const MetalPrivateDevice *device,
                         const RenderParameters &userRenderParams,
                         const VertexFormat &vertexFormat,
                         const std::string &shaderSource)
    : Shader(userRenderParams), _name(name) {
    load(device, vertexFormat, shaderSource);
}

MetalShader::~MetalShader() { clear(); }

MetalPrivateLibrary *MetalShader::library() const { return _library; }

MetalPrivateFunction *MetalShader::vertexFunction() const { return _vertFunc; }

MetalPrivateFunction *MetalShader::fragmentFunction() const {
    return _fragFunc;
}

const std::string &MetalShader::name() const { return _name; }

void MetalShader::onBind(const Renderer *renderer) {
    UNUSED_VARIABLE(renderer);
}

void MetalShader::onUnbind(const Renderer *renderer) {
    UNUSED_VARIABLE(renderer);
}

void MetalShader::clear() {
    if (_library != nullptr) {
        delete _library;
        _library = nullptr;
    }

    if (_vertFunc != nullptr) {
        delete _vertFunc;
        _vertFunc = nullptr;
    }

    if (_fragFunc != nullptr) {
        delete _fragFunc;
        _fragFunc = nullptr;
    }
}

void MetalShader::load(const MetalPrivateDevice *device,
                       const VertexFormat &vertexFormat,
                       const std::string &shaderSource) {
    _vertexFormat = vertexFormat;

    mtlpp::Device d = device->value;

    ns::Error error(ns::Handle{.ptr = nullptr});
    _library = new MetalPrivateLibrary(
        d.NewLibrary(shaderSource.c_str(), mtlpp::CompileOptions(), &error));

    if (error.GetPtr() != nullptr) {
        ns::String errorDesc = error.GetLocalizedDescription();
        JET_ERROR << errorDesc.GetCStr();
    }

    _vertFunc =
        new MetalPrivateFunction(_library->value.NewFunction("vertFunc"));
    _fragFunc =
        new MetalPrivateFunction(_library->value.NewFunction("fragFunc"));
}

}  // namespace gfx
}  // namespace jet
