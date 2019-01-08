// Copyright (c) 2018 Doyub Kim
//
// I am making my contributions/submissions to this project solely in my
// personal capacity and am not conveying any rights to any intellectual
// property of any third parties.

#ifndef INCLUDE_JET_GFX_METAL_RENDERER_H_
#define INCLUDE_JET_GFX_METAL_RENDERER_H_

#include <jet/macros.h>

#ifdef JET_MACOSX

#include <jet.gfx/renderer.h>

namespace jet {

namespace gfx {

class MetalWindow;
class MetalPrivateDevice;
class MetalPrivateCommandBuffer;
class MetalPrivateCommandQueue;
class MetalPrivateRenderCommandEncoder;
class MetalPrivateRenderPassDescriptor;
class MetalPrivateRenderPipelineState;
class MetalShader;

using MetalShaderPtr = std::shared_ptr<MetalShader>;

class MetalRenderer final : public Renderer {
 public:
    JET_NON_COPYABLE(MetalRenderer);

    MetalRenderer(MetalWindow* window);

    ~MetalRenderer();

    //!
    //! Creates a vertex buffer with given parameters.
    //!
    //! \param shader Shader object for the buffer.
    //! \param vertices Vertex data.
    //! \param numberOfVertices Number of vertices.
    //! \return New vertex buffer.
    //!
    VertexBufferPtr createVertexBuffer(const ShaderPtr& shader,
                                       const float* vertexData,
                                       size_t numberOfVertices) override;

    //!
    //! Creates an index buffer with given parameters.
    //!
    //! \param vertexBuffer Vertices for the index buffer.
    //! \param indices Index data.
    //! \param numberOfIndices Number of indices.
    //! \return New index buffer.
    //!
    IndexBufferPtr createIndexBuffer(const VertexBufferPtr& vertexBuffer,
                                     const uint32_t* indices,
                                     size_t numberOfIndices) override;

    //!
    //! Creates a 2-D texture with 8-bit image and given parameters.
    //!
    //! \param data 8-bit texture image data.
    //! \param size Size of the data.
    //! \return New 2-D texture.
    //!
    Texture2Ptr createTexture2(const ConstArrayView2<Vector4B>& data) override;

    //!
    //! Creates a 2-D texture with 32-bit image and given parameters.
    //!
    //! \param data 32-bit texture image data.
    //! \param size Size of the data.
    //! \return New 2-D texture.
    //!
    Texture2Ptr createTexture2(const ConstArrayView2<Vector4F>& data) override;

    //!
    //! Creates a 3-D texture with 8-bit image and given parameters.
    //!
    //! \param data 8-bit texture image data.
    //! \param size Size of the data.
    //! \return New 3-D texture.
    //!
    Texture3Ptr createTexture3(const ConstArrayView3<Vector4B>& data) override;

    //!
    //! Creates a 3-D texture with 32-bit image and given parameters.
    //!
    //! \param data 32-bit texture image data.
    //! \param size Size of the data.
    //! \return New 3-D texture.
    //!
    Texture3Ptr createTexture3(const ConstArrayView3<Vector4F>& data) override;

    //!
    //! Creates a shader object with given preset shader name.
    //!
    //! \param shaderName Preset shader name.
    //! \return New shader.
    //!
    ShaderPtr createPresetShader(const std::string& shaderName) const override;

    //!
    //! Sets current render primitive type state for drawing.
    //!
    //! \param type Primitive type.
    //!
    void setPrimitiveType(PrimitiveType type) override;

    //!
    //! Draws currently bound object.
    //!
    //! \param numberOfVertices Number of vertices.
    //!
    void draw(size_t numberOfVertices) override;

    //!
    //! Draws currently bound indexed object.
    //!
    //! \param numberOfIndices Number of indices.
    //!
    void drawIndexed(size_t numberOfIndices) override;

    MetalPrivateDevice* device() const;

    MetalPrivateCommandQueue* commandQueue() const;

    MetalPrivateRenderCommandEncoder* renderCommandEncoder() const;

    MetalPrivateRenderPipelineState* findRenderPipelineState(
        const std::string& name) const;

 protected:
    //! Called when rendering a frame begins.
    void onRenderBegin() override;

    //! Called when rendering a frame ended.
    void onRenderEnd() override;

    //! Called when the view has resized.
    void onResize(const Viewport& viewport) override;

    //! Called when the render states has changed.
    void onSetRenderStates(const RenderStates& states) override;

 private:
    MetalWindow* _window = nullptr;

    std::unique_ptr<MetalPrivateDevice> _device;
    std::unique_ptr<MetalPrivateCommandQueue> _commandQueue;

    std::unique_ptr<MetalPrivateCommandBuffer> _commandBuffer;
    std::unique_ptr<MetalPrivateRenderPassDescriptor> _renderPassDescriptor;
    std::unique_ptr<MetalPrivateRenderCommandEncoder> _renderCommandEncoder;

    PrimitiveType _primitiveType = PrimitiveType::Triangles;

    mutable std::unordered_map<std::string,
                               std::unique_ptr<MetalPrivateRenderPipelineState>>
        _renderPipelineStates;

    void onClearRenderables() override;

    MetalPrivateRenderPipelineState* createRenderPipelineStateFromShader(
        const MetalShaderPtr& shader) const;

    // Applies given render states to render command encoder.
    void applyCurrentRenderStatesToDevice();
};

using MetalRendererPtr = std::shared_ptr<MetalRenderer>;

}  // namespace gfx

}  // namespace jet

#endif  // INCLUDE_JET_GFX_METAL_RENDERER_H_

#endif  // JET_MACOSX
