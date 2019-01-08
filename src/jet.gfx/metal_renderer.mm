// Copyright (c) 2018 Doyub Kim
//
// I am making my contributions/submissions to this project solely in my
// personal capacity and am not conveying any rights to any intellectual
// property of any third parties.

#undef JET_USE_GL
#include <common.h>

#include <metal_preset_shaders.h>
#include <metal_view.h>
#include <mtlpp_wrappers.h>

#include <jet.gfx/metal_renderer.h>
#include <jet.gfx/metal_shader.h>
#include <jet.gfx/metal_vertex_buffer.h>
#include <jet.gfx/metal_window.h>

#import <MetalKit/MetalKit.h>

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

mtlpp::BlendFactor convertBlendFactor(RenderStates::BlendFactor blendFactor) {
    switch (blendFactor) {
        case RenderStates::BlendFactor::Zero:
            return mtlpp::BlendFactor::Zero;
        case RenderStates::BlendFactor::One:
            return mtlpp::BlendFactor::One;
        case RenderStates::BlendFactor::SrcAlpha:
            return mtlpp::BlendFactor::SourceAlpha;
        case RenderStates::BlendFactor::OneMinusSrcAlpha:
            return mtlpp::BlendFactor::OneMinusSourceAlpha;
        case RenderStates::BlendFactor::SrcColor:
            return mtlpp::BlendFactor::SourceColor;
        case RenderStates::BlendFactor::OneMinusSrcColor:
            return mtlpp::BlendFactor::OneMinusSourceColor;
        case RenderStates::BlendFactor::DestAlpha:
            return mtlpp::BlendFactor::DestinationAlpha;
        case RenderStates::BlendFactor::OneMinusDestAlpha:
            return mtlpp::BlendFactor::OneMinusDestinationAlpha;
        case RenderStates::BlendFactor::DestColor:
            return mtlpp::BlendFactor::OneMinusDestinationColor;
        case RenderStates::BlendFactor::OneMinusDestColor:
            return mtlpp::BlendFactor::OneMinusDestinationColor;
        default:
            JET_ASSERT(false);
            return mtlpp::BlendFactor::Zero;
    }
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

}  // namespace

MetalRenderer::MetalRenderer(MetalWindow* window) : _window(window) {
    // Create device
    _device = std::make_unique<MetalPrivateDevice>(
        mtlpp::Device::CreateSystemDefaultDevice());

    // Create command queue
    _commandQueue = std::make_unique<MetalPrivateCommandQueue>(
        _device->value.NewCommandQueue());

    // Initialize render state
    RenderStates defaultRenderStates;
    setRenderStates(defaultRenderStates);
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
        shader->_vertexUniformSize = sizeof(SimpleColorVertexUniforms);
    } else if (shaderName == "points") {
        params.add("Radius", 1.f);

        shader = std::make_shared<MetalShader>(
            shaderName, _device.get(), params, VertexFormat::Position3Color4,
            kPointsShaders);
        shader->_vertexUniformSize = sizeof(PointsVertexUniforms);
    }

    if (shader) {
        auto iter = _renderPipelineStates.find(shaderName);
        if (iter == _renderPipelineStates.end()) {
            auto renderPipelineState =
                createRenderPipelineStateFromShader(shader);
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
        _renderPassDescriptor->value.GetColorAttachments()[0].SetLoadAction(
            mtlpp::LoadAction::Clear);
        _renderPassDescriptor->value.GetColorAttachments()[0].SetStoreAction(
            mtlpp::StoreAction::DontCare);
        _renderPassDescriptor->value.GetColorAttachments()[0].SetClearColor(
            mtlpp::ClearColor(bg.x, bg.y, bg.z, bg.w));

        _renderPassDescriptor->value.GetDepthAttachment().SetLoadAction(
            mtlpp::LoadAction::Clear);
        _renderPassDescriptor->value.GetDepthAttachment().SetStoreAction(
            mtlpp::StoreAction::DontCare);
        _renderPassDescriptor->value.GetDepthAttachment().SetClearDepth(1.0);

        _renderCommandEncoder =
            std::make_unique<MetalPrivateRenderCommandEncoder>(
                _commandBuffer->value.RenderCommandEncoder(
                    _renderPassDescriptor->value));

        // Set render state
        applyCurrentRenderStatesToDevice();
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
    JET_ASSERT(_renderCommandEncoder);

    mtlpp::Viewport mtlViewport{.OriginX = viewport.x,
                                .OriginY = viewport.y,
                                .Width = viewport.width,
                                .Height = viewport.height,
                                .ZNear = camera()->state.nearClipPlane,
                                .ZFar = camera()->state.farClipPlane};
    _renderCommandEncoder->value.SetViewport(mtlViewport);
}

void MetalRenderer::onSetRenderStates(const RenderStates& states) {
    UNUSED_VARIABLE(states);
}

void MetalRenderer::onClearRenderables() { _renderPipelineStates.clear(); }

MetalPrivateRenderPipelineState*
MetalRenderer::createRenderPipelineStateFromShader(
    const MetalShaderPtr& shader) const {
    mtlpp::RenderPipelineDescriptor renderPipelineDesc;
    renderPipelineDesc.SetLabel(ns::String(shader->name().c_str()));
    renderPipelineDesc.SetVertexFunction(shader->vertexFunction()->value);
    renderPipelineDesc.SetFragmentFunction(shader->fragmentFunction()->value);
    renderPipelineDesc.SetVertexDescriptor(
        createVertexDescripter(shader->vertexFormat(),
                               /* bufferIndex*/ 0));
    auto colorAttachment0 = renderPipelineDesc.GetColorAttachments()[0];
    colorAttachment0.SetPixelFormat(mtlpp::PixelFormat::BGRA8Unorm);
    renderPipelineDesc.SetDepthAttachmentPixelFormat(
        mtlpp::PixelFormat::Depth32Float);

    if (renderStates().isBlendEnabled) {
        colorAttachment0.SetBlendingEnabled(true);
        colorAttachment0.SetRgbBlendOperation(mtlpp::BlendOperation::Add);
        colorAttachment0.SetAlphaBlendOperation(mtlpp::BlendOperation::Add);
        auto sourceFactor =
            convertBlendFactor(renderStates().sourceBlendFactor);
        auto dstFactor =
            convertBlendFactor(renderStates().destinationBlendFactor);
        colorAttachment0.SetSourceRgbBlendFactor(sourceFactor);
        colorAttachment0.SetSourceAlphaBlendFactor(sourceFactor);
        colorAttachment0.SetDestinationRgbBlendFactor(dstFactor);
        colorAttachment0.SetDestinationAlphaBlendFactor(dstFactor);
    } else {
        colorAttachment0.SetBlendingEnabled(false);
    }

    JET_DEBUG << "Metal render pipeline state created with shader "
              << shader->name();

    // Retrieve uniform info
    mtlpp::RenderPipelineReflection reflectionObj;
    auto options =
        mtlpp::PipelineOption(int(mtlpp::PipelineOption::BufferTypeInfo) |
                              int(mtlpp::PipelineOption::ArgumentInfo));

    ns::Error error(ns::Handle{.ptr = nullptr});
    mtlpp::RenderPipelineState pso = _device->value.NewRenderPipelineState(
        renderPipelineDesc, options, &reflectionObj, &error);

    if (error) {
        JET_ERROR << error.GetLocalizedDescription().GetCStr();
    } else {
        auto vertexArguments = reflectionObj.GetVertexArguments();
        for (uint32_t i = 0; i < vertexArguments.GetSize(); ++i) {
            mtlpp::Argument arg = vertexArguments[i];

            if (arg.GetBufferDataType() == mtlpp::DataType::Struct &&
                arg.GetBufferStructType()) {
                auto members = arg.GetBufferStructType().GetMembers();
                for (uint32_t j = 0; j < members.GetSize(); ++j) {
                    mtlpp::StructMember uniform = members[j];

                    std::string name = uniform.GetName().GetCStr();
                    if (shader->userRenderParameters().has(name)) {
                        shader->_vertexUniformLocations[name] =
                            uniform.GetOffset();
                    }
                }
            }
        }
    }

    return new MetalPrivateRenderPipelineState(
        _device->value.NewRenderPipelineState(renderPipelineDesc, nullptr));
}

void MetalRenderer::applyCurrentRenderStatesToDevice() {
    JET_ASSERT(_renderCommandEncoder);

    switch (renderStates().cullMode) {
        case RenderStates::CullMode::None:
            _renderCommandEncoder->value.SetCullMode(mtlpp::CullMode::None);
            break;
        case RenderStates::CullMode::Front:
            _renderCommandEncoder->value.SetCullMode(mtlpp::CullMode::Front);
            break;
        case RenderStates::CullMode::Back:
            _renderCommandEncoder->value.SetCullMode(mtlpp::CullMode::Back);
            break;
    }

    if (renderStates().isFrontFaceClockWise) {
        _renderCommandEncoder->value.SetFrontFacingWinding(
            mtlpp::Winding::Clockwise);
    } else {
        _renderCommandEncoder->value.SetFrontFacingWinding(
            mtlpp::Winding::CounterClockwise);
    }

    mtlpp::DepthStencilDescriptor depthStencilDesc;
    if (renderStates().isDepthTestEnabled) {
        depthStencilDesc.SetDepthCompareFunction(mtlpp::CompareFunction::Less);
        depthStencilDesc.SetDepthWriteEnabled(true);
    } else {
        depthStencilDesc.SetDepthWriteEnabled(false);
    }

    mtlpp::DepthStencilState depthStencilState =
        _device->value.NewDepthStencilState(depthStencilDesc);
    _renderCommandEncoder->value.SetDepthStencilState(depthStencilState);
}
}
}
