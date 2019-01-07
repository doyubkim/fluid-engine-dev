// Copyright (c) 2018 Doyub Kim
//
// I am making my contributions/submissions to this project solely in my
// personal capacity and am not conveying any rights to any intellectual
// property of any third parties.

#include <common.h>

#ifdef JET_MACOSX

#include "metal_preset_shaders.h"
#include "mtlpp_wrappers.h"

#include <jet.gfx/metal_renderer.h>
#include <jet.gfx/metal_shader.h>

#include <simd/simd.h>

namespace jet {
namespace gfx {

namespace {

struct PointsVertexUniforms {
    simd_float4x4 ModelViewProjection;
    float Radius;
};

}  // namespace

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
    const auto mtlRenderer = dynamic_cast<const MetalRenderer *>(renderer);
    JET_ASSERT(mtlRenderer != nullptr);

    auto renderPipelineState = mtlRenderer->findRenderPipelineState(name());
    JET_ASSERT(renderPipelineState != nullptr);

    auto commandEncoder = mtlRenderer->renderCommandEncoder();

    commandEncoder->value.SetRenderPipelineState(renderPipelineState->value);

    // Load default parameters
    Array1<uint8_t> paramData(_vertexUniformSize);
//    JET_INFO << _vertexUniformSize;

    const auto &camera = renderer->camera();
    const Matrix4x4F glToMetal(
        {{1, 0, 0, 0}, {0, 1, 0, 0}, {0, 0, 0.5f, 0.5f}, {0, 0, 0, 1}});
    const Matrix4x4F matrix =
        glToMetal * camera->projectionMatrix() * camera->state.viewMatrix();
    auto row1 = simd_make_float4(matrix(0, 0), matrix(0, 1), matrix(0, 2),
                                 matrix(0, 3));
    auto row2 = simd_make_float4(matrix(1, 0), matrix(1, 1), matrix(1, 2),
                                 matrix(1, 3));
    auto row3 = simd_make_float4(matrix(2, 0), matrix(2, 1), matrix(2, 2),
                                 matrix(2, 3));
    auto row4 = simd_make_float4(matrix(3, 0), matrix(3, 1), matrix(3, 2),
                                 matrix(3, 3));
    auto mvp = simd_matrix_from_rows(row1, row2, row3, row4);
    memcpy(paramData.data(), &mvp, sizeof(mvp));

    // Apply parameters
    const auto &userParams = userRenderParameters();
    for (auto name : userParams.names()) {
        auto iter = _vertexUniformLocations.find(name);
        if (iter != _vertexUniformLocations.end()) {
            const int32_t *data = userParams.buffer(name);
            const size_t size = RenderParameters::typeSizeInBytes(
                userParams.metadata(name).type);
//            const float *floatData = reinterpret_cast<const float *>(data);
//            JET_INFO << name << ", " << floatData[0];
            memcpy(paramData.data() + iter->second, data, size);
        }
    }
    //
    //    PointsVertexUniforms tmp;
    //    memcpy(&tmp, paramData.data(), paramData.length());
    //    JET_INFO << tmp.Radius;

    commandEncoder->value.SetVertexData(paramData.data(),
                                        (uint32_t)paramData.length(), 1);
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

#endif  // JET_MACOSX
