// Copyright (c) 2018 Doyub Kim
//
// I am making my contributions/submissions to this project solely in my
// personal capacity and am not conveying any rights to any intellectual
// property of any third parties.

#undef JET_USE_GL
#import <common.h>

#import <metal_preset_shaders.h>
#import <metal_view.h>
#import <mtlpp_wrappers.h>

#import <jet.gfx/metal_renderer.h>
#import <jet.gfx/metal_shader.h>
#import <jet.gfx/metal_vertex_buffer.h>
#import <jet.gfx/metal_window.h>

#import <MetalKit/MetalKit.h>

// TODO: Test
#import <jet.gfx/points_renderable.h>

namespace jet {
namespace gfx {

namespace {

mtlpp::Drawable getCurrentDrawable(const MetalWindow* window) {
    return ns::Handle{
        (__bridge void*)((__bridge MTKView*)window->view()->GetPtr())
            .currentDrawable};
}

mtlpp::RenderPassDescriptor getCurrentRenderPassDescriptor(
    const MetalWindow* window) {
    return ns::Handle{
        (__bridge void*)((__bridge MTKView*)window->view()->GetPtr())
            .currentRenderPassDescriptor};
}

mtlpp::VertexDescriptor createVertexDescripter(VertexFormat vertexFormat,
                                               uint32_t bufferIndex) {
    mtlpp::VertexDescriptor desc;
    const MTLVertexDescriptor* vertexDescriptor =
        (const MTLVertexDescriptor*)(desc.GetPtr());

    uint32_t offset = 0;
    int attributeId = 0;

    if (static_cast<int>(vertexFormat & VertexFormat::Position3)) {
        size_t numberOfFloats =
            VertexHelper::getNumberOfFloats(VertexFormat::Position3);
        vertexDescriptor.attributes[attributeId].format = MTLVertexFormatFloat3;
        vertexDescriptor.attributes[attributeId].bufferIndex = bufferIndex;
        vertexDescriptor.attributes[attributeId].offset = offset;
        offset += numberOfFloats * sizeof(float);
        ++attributeId;
    }

    if (static_cast<int>(vertexFormat & VertexFormat::Normal3)) {
        size_t numberOfFloats =
            VertexHelper::getNumberOfFloats(VertexFormat::Normal3);
        vertexDescriptor.attributes[attributeId].format = MTLVertexFormatFloat3;
        vertexDescriptor.attributes[attributeId].bufferIndex = bufferIndex;
        vertexDescriptor.attributes[attributeId].offset = offset;
        offset += numberOfFloats * sizeof(float);
        ++attributeId;
    }

    if (static_cast<int>(vertexFormat & VertexFormat::TexCoord2)) {
        size_t numberOfFloats =
            VertexHelper::getNumberOfFloats(VertexFormat::TexCoord2);
        vertexDescriptor.attributes[attributeId].format = MTLVertexFormatFloat2;
        vertexDescriptor.attributes[attributeId].bufferIndex = bufferIndex;
        vertexDescriptor.attributes[attributeId].offset = offset;
        offset += numberOfFloats * sizeof(float);
        ++attributeId;
    }

    if (static_cast<int>(vertexFormat & VertexFormat::TexCoord3)) {
        size_t numberOfFloats =
            VertexHelper::getNumberOfFloats(VertexFormat::TexCoord3);
        vertexDescriptor.attributes[attributeId].format = MTLVertexFormatFloat3;
        vertexDescriptor.attributes[attributeId].bufferIndex = bufferIndex;
        vertexDescriptor.attributes[attributeId].offset = offset;
        offset += numberOfFloats * sizeof(float);
        ++attributeId;
    }

    if (static_cast<int>(vertexFormat & VertexFormat::Color4)) {
        size_t numberOfFloats =
            VertexHelper::getNumberOfFloats(VertexFormat::Color4);
        vertexDescriptor.attributes[attributeId].format = MTLVertexFormatFloat4;
        vertexDescriptor.attributes[attributeId].bufferIndex = bufferIndex;
        vertexDescriptor.attributes[attributeId].offset = offset;
        offset += numberOfFloats * sizeof(float);
        ++attributeId;
    }

    vertexDescriptor.layouts[0].stride = offset;

    return desc;
}

mtlpp::PrimitiveType convertPrimitiveType(PrimitiveType primitiveType) {
    if (primitiveType == PrimitiveType::Points) {
        return mtlpp::PrimitiveType::Point;
    } else if (primitiveType == PrimitiveType::Lines) {
        return mtlpp::PrimitiveType::Line;
    } else if (primitiveType == PrimitiveType::LineStrip) {
        return mtlpp::PrimitiveType::LineStrip;
    } else if (primitiveType == PrimitiveType::Triangles) {
        return mtlpp::PrimitiveType::Triangle;
    } else if (primitiveType == PrimitiveType::TriangleStrip) {
        return mtlpp::PrimitiveType::TriangleStrip;
    }

    throw std::invalid_argument("Unknown primitive type given");
}

MetalPrivateRenderPipelineState* createRenderPipelineStateFromShader(
    MetalPrivateDevice* device, MetalShaderPtr shader) {
    mtlpp::RenderPipelineDescriptor renderPipelineDesc;
    renderPipelineDesc.SetLabel(ns::String(shader->name().c_str()));
    renderPipelineDesc.SetVertexFunction(shader->vertexFunction()->value);
    renderPipelineDesc.SetFragmentFunction(shader->fragmentFunction()->value);
    renderPipelineDesc.SetVertexDescriptor(
        createVertexDescripter(shader->vertexFormat(),
                               /* bufferIndex*/ 0));
    renderPipelineDesc.GetColorAttachments()[0].SetPixelFormat(
        mtlpp::PixelFormat::BGRA8Unorm);

    JET_DEBUG << "Metal render pipeline state created with shader "
              << shader->name();

    return new MetalPrivateRenderPipelineState(
        device->value.NewRenderPipelineState(renderPipelineDesc, nullptr));
}

}  // namespace

MetalRenderer::MetalRenderer(MetalWindow* window) : _window(window) {
    // Create device
    _device = std::make_unique<MetalPrivateDevice>(
        mtlpp::Device::CreateSystemDefaultDevice());

    // Create command queue
    _commandQueue = std::make_unique<MetalPrivateCommandQueue>(
        _device->value.NewCommandQueue());
}

MetalRenderer::~MetalRenderer() {}

VertexBufferPtr MetalRenderer::createVertexBuffer(const ShaderPtr& shader,
                                                  const float* vertexData,
                                                  size_t numberOfVertices) {
    auto ret = std::make_shared<MetalVertexBuffer>(_device.get());
    ret->resize(shader, vertexData, numberOfVertices);
    return ret;
}

IndexBufferPtr MetalRenderer::createIndexBuffer(
    const VertexBufferPtr& vertexBuffer, const uint32_t* indices,
    size_t numberOfIndices) {
    throw NotImplementedException(
        "MetalRenderer::createIndexBuffer not implemented");
    return nullptr;
}

Texture2Ptr MetalRenderer::createTexture2(
    const ConstArrayView2<Vector4B>& data) {
    throw NotImplementedException(
        "MetalRenderer::createTexture2 not implemented");
    return nullptr;
}

Texture2Ptr MetalRenderer::createTexture2(
    const ConstArrayView2<Vector4F>& data) {
    throw NotImplementedException(
        "MetalRenderer::createTexture2 not implemented");
    return nullptr;
}

Texture3Ptr MetalRenderer::createTexture3(
    const ConstArrayView3<Vector4B>& data) {
    throw NotImplementedException(
        "MetalRenderer::createTexture3 not implemented");
    return nullptr;
}

Texture3Ptr MetalRenderer::createTexture3(
    const ConstArrayView3<Vector4F>& data) {
    throw NotImplementedException(
        "MetalRenderer::createTexture3 not implemented");
    return nullptr;
}

ShaderPtr MetalRenderer::createPresetShader(
    const std::string& shaderName) const {
    RenderParameters params;
    MetalShaderPtr shader;

    if (shaderName == "simple_color") {
        shader = std::make_shared<MetalShader>(
            shaderName, _device.get(), params, VertexFormat::Position3Color4,
            kSimpleColorShader);
    } else if (shaderName == "points") {
        params.add("Radius", 1.f);

        shader = std::make_shared<MetalShader>(
            shaderName, _device.get(), params, VertexFormat::Position3Color4,
            kPointsShaders);
    }

    if (shader) {
        auto iter = _renderPipelineStates.find(shaderName);
        if (iter == _renderPipelineStates.end()) {
            auto renderPipelineState =
                createRenderPipelineStateFromShader(_device.get(), shader);
            _renderPipelineStates[shaderName] =
                std::unique_ptr<MetalPrivateRenderPipelineState>(
                    renderPipelineState);
        }
    }

    return shader;
}

void MetalRenderer::setPrimitiveType(PrimitiveType type) {
    _primitiveType = type;
}

void MetalRenderer::draw(size_t numberOfVertices) {
    _renderCommandEncoder->value.Draw(convertPrimitiveType(_primitiveType),
                                      /* vertexStart */ 0,
                                      /* vertexCount */ numberOfVertices);
}

void MetalRenderer::drawIndexed(size_t numberOfIndices) {
    throw NotImplementedException(
        "MetalRenderer::drawIndexed is not implemented, yet");
}

//void MetalRenderer::render() {
//    onRenderBegin();
//
//    if (_renderPassDescriptor) {
//        // For each renderable...
////        bindShader(g_shader);
////
////        bindVertexBuffer(g_vertexBuffer);
////
////        setPrimitiveType(PrimitiveType::Points);
////
////        draw(g_vertexBuffer->numberOfVertices());
//        g_renderable->render(this);
//    }
//
//    onRenderEnd();
//}

MetalPrivateDevice* MetalRenderer::device() const { return _device.get(); }

MetalPrivateCommandQueue* MetalRenderer::commandQueue() const {
    return _commandQueue.get();
}

MetalPrivateRenderCommandEncoder* MetalRenderer::renderCommandEncoder() const {
    return _renderCommandEncoder.get();
}

MetalPrivateRenderPipelineState* MetalRenderer::findRenderPipelineState(
    const std::string& name) const {
    auto iter = _renderPipelineStates.find(name);
    if (iter != _renderPipelineStates.end()) {
        return iter->second.get();
    } else {
        return nullptr;
    }
}

void MetalRenderer::onRenderBegin() {
    _commandBuffer = std::make_unique<MetalPrivateCommandBuffer>(
        _commandQueue->value.CommandBuffer());
    _renderPassDescriptor = std::make_unique<MetalPrivateRenderPassDescriptor>(
        getCurrentRenderPassDescriptor(_window));
    if (_renderPassDescriptor) {
        const auto& bg = backgroundColor();
        _renderPassDescriptor->value.GetColorAttachments()[0].SetClearColor(
            mtlpp::ClearColor(bg.x, bg.y, bg.z, bg.w));

        _renderCommandEncoder =
            std::make_unique<MetalPrivateRenderCommandEncoder>(
                _commandBuffer->value.RenderCommandEncoder(
                    _renderPassDescriptor->value));
    }
}

void MetalRenderer::onRenderEnd() {
    if (_renderPassDescriptor) {
        _renderCommandEncoder->value.EndEncoding();
        _commandBuffer->value.Present(getCurrentDrawable(_window));
    }
    _commandBuffer->value.Commit();
    _commandBuffer->value.WaitUntilCompleted();
}

void MetalRenderer::onResize(const Viewport& viewport) {
    // TODO
}

void MetalRenderer::onSetRenderStates(const RenderStates& states) {
    // TODO
}
}
}
