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

namespace jet {
namespace gfx {

namespace {

MetalShaderPtr g_shader;
MetalVertexBufferPtr g_vertexBuffer;

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

    NSLog(@"%lu", ((__bridge MTLVertexDescriptor*)renderPipelineDesc
            .GetVertexDescriptor()
            .GetPtr())
            .layouts[0]
            .stride);

    JET_INFO << "Metal render pipeline state created with shader "
             << shader->name();

    return new MetalPrivateRenderPipelineState(
            device->value.NewRenderPipelineState(renderPipelineDesc, nullptr));
}

}  // namespace

MetalRenderer::MetalRenderer(MetalWindow* window) : _window(window) {
    // Create device
    _device =
            new MetalPrivateDevice(mtlpp::Device::CreateSystemDefaultDevice());

    // Create command queue
    _commandQueue =
            new MetalPrivateCommandQueue(_device->value.NewCommandQueue());

    // TEMP
    g_shader = std::static_pointer_cast<MetalShader>(
            createPresetShader("simple_color"));

    const float vertexData[] = {0.0f,  1.0f,  0.0f, 1.0f, 0.0f, 0.0f, 1.0f,
                                -1.0f, -1.0f, 0.0f, 0.0f, 1.0f, 0.0f, 1.0f,
                                1.0f,  -1.0f, 0.0f, 0.0f, 0.0f, 1.0f, 1.0f};

    g_vertexBuffer =
            std::make_shared<MetalVertexBuffer>(_device, g_shader, ConstArrayView1<float>(vertexData, 3));
}

MetalRenderer::~MetalRenderer() {
    delete _device;
    delete _commandQueue;

    for (auto iter : _renderPipelineStates) {
        delete iter.second;
    }
}

VertexBufferPtr MetalRenderer::createVertexBuffer(const ShaderPtr& shader,
                                                  const ConstArrayView1<float>& vertices) {
    return nullptr;
}

IndexBufferPtr MetalRenderer::createIndexBuffer(
        const VertexBufferPtr& vertexBuffer, const ConstArrayView1<uint32_t>& indices) {
    return nullptr;
}

Texture2Ptr MetalRenderer::createTexture2(
        const ConstArrayView2<Vector4B>& data) {
    return nullptr;
}

Texture2Ptr MetalRenderer::createTexture2(const ConstArrayView2<Vector4F>& data) {
    return nullptr;
}

Texture3Ptr MetalRenderer::createTexture3(
        const ConstArrayView3<Vector4B>& data) {
    return nullptr;
}

Texture3Ptr MetalRenderer::createTexture3(const ConstArrayView3<Vector4F>& data) {
    return nullptr;
}

ShaderPtr MetalRenderer::createPresetShader(
        const std::string& shaderName) const {
    RenderParameters params;
    MetalShaderPtr shader;

    if (shaderName == "simple_color") {
        shader = std::make_shared<MetalShader>(shaderName, _device, params,
                                               VertexFormat::Position3Color4,
                                               kSimpleColorShader);
    }

    if (shader) {
        auto iter = _renderPipelineStates.find(shaderName);
        if (iter == _renderPipelineStates.end()) {
            auto renderPipelineState =
                    createRenderPipelineStateFromShader(_device, shader);
            _renderPipelineStates[shaderName] = renderPipelineState;
        }
    }

    return shader;
}

void MetalRenderer::setPrimitiveType(PrimitiveType type) {}

void MetalRenderer::draw(size_t numberOfVertices) {}

void MetalRenderer::drawIndexed(size_t numberOfIndices) {}

void MetalRenderer::render() {
    mtlpp::CommandBuffer commandBuffer = _commandQueue->value.CommandBuffer();

    mtlpp::RenderPassDescriptor renderPassDesc =
            getCurrentRenderPassDescriptor(_window);
    if (renderPassDesc) {
        mtlpp::RenderCommandEncoder renderCommandEncoder =
                commandBuffer.RenderCommandEncoder(renderPassDesc);

        // For each renderable...
        auto iter = _renderPipelineStates.find(g_shader->name());
        if (iter != _renderPipelineStates.end()) {
            // renderer->bindShader(_shader);
            auto renderPipelineState = iter->second;
            renderCommandEncoder.SetRenderPipelineState(
                    renderPipelineState->value);

            // renderer->bindVertexBuffer(_vertexBuffer);
            renderCommandEncoder.SetVertexBuffer(
                    g_vertexBuffer->buffer()->value, /* offset */ 0,
                    /* buffer0 */ 0);

            // renderer->setPrimitiveType(PrimitiveType::Triangles);
            mtlpp::PrimitiveType primitiveType = mtlpp::PrimitiveType::Triangle;

            // renderer->draw(_vertexBuffer->numberOfVertices());
            renderCommandEncoder.Draw(primitiveType, /* vertexStart */ 0,
                    /* vertexCount */ 3);
        }

        renderCommandEncoder.EndEncoding();
        commandBuffer.Present(getCurrentDrawable(_window));
    }

    commandBuffer.Commit();
    commandBuffer.WaitUntilCompleted();
}

MetalPrivateDevice* MetalRenderer::device() const { return _device; }

MetalPrivateCommandQueue* MetalRenderer::commandQueue() const {
    return _commandQueue;
}

void MetalRenderer::onRenderBegin() {}

void MetalRenderer::onRenderEnd() {}

void MetalRenderer::onResize(const Viewport& viewport) {}

void MetalRenderer::onSetRenderStates(const RenderStates& states) {}

}
}
