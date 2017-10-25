// Copyright (c) 2017 Doyub Kim
//
// I am making my contributions/submissions to this project solely in my
// personal capacity and am not conveying any rights to any intellectual
// property of any third parties.

#ifndef INCLUDE_JET_VIZ_RENDERER_H_
#define INCLUDE_JET_VIZ_RENDERER_H_

#include <jet.viz/camera.h>
#include <jet.viz/color.h>
#include <jet.viz/index_buffer.h>
#include <jet.viz/render_states.h>
#include <jet.viz/renderable.h>
#include <jet.viz/shader.h>
#include <jet.viz/texture2.h>
#include <jet.viz/texture3.h>
#include <jet.viz/vertex_buffer.h>
#include <jet.viz/viewport.h>

#include <map>
#include <string>
#include <vector>

namespace jet { namespace viz {

class Renderer {
 public:
    Renderer();
    virtual ~Renderer();

    virtual VertexBufferPtr createVertexBuffer(const ShaderPtr& shader,
                                               const float* vertices,
                                               std::size_t numberOfPoints) = 0;

    virtual IndexBufferPtr createIndexBuffer(
        const VertexBufferPtr& vertexBuffer, const std::uint32_t* indices,
        std::size_t numberOfIndices) = 0;

    virtual Texture2Ptr createTexture2(const std::uint8_t* const data,
                                       const Size2& size) = 0;

    virtual Texture2Ptr createTexture2(const float* const data,
                                       const Size2& size) = 0;

    virtual Texture3Ptr createTexture3(const float* const data,
                                       const Size3& size) = 0;

    virtual ShaderPtr createPresetShader(
        const std::string& shaderName) const = 0;

    virtual void setPrimitiveType(PrimitiveType type) = 0;

    virtual void draw(std::size_t numberOfVertices) = 0;

    virtual void drawIndexed(std::size_t numberOfIndices) = 0;

    void render();
    void resize(const Viewport& viewport);
    void addRenderable(const RenderablePtr& renderable);
    void clearRenderables();

    void bindShader(const ShaderPtr& shader);
    void unbindShader(const ShaderPtr& shader);
    void bindVertexBuffer(const VertexBufferPtr& vertexBuffer);
    void unbindVertexBuffer(const VertexBufferPtr& vertexBuffer);
    void bindIndexBuffer(const IndexBufferPtr& indexBuffer);
    void unbindIndexBuffer(const IndexBufferPtr& indexBuffer);
    void bindTexture(const Texture2Ptr& texture, unsigned int slotId);
    void bindTexture(const Texture3Ptr& texture, unsigned int slotId);

    const Viewport& viewport() const;

    const CameraPtr& camera() const;
    void setCamera(const CameraPtr& camera);

    const RenderStates& renderStates() const;
    void setRenderStates(const RenderStates& states);

    const Color& backgroundColor() const;
    void setBackgroundColor(const Color& color);

 protected:
    virtual void onRenderBegin() = 0;
    virtual void onRenderEnd() = 0;
    virtual void onResize(const Viewport& viewport) = 0;
    virtual void onSetRenderStates(const RenderStates& states) = 0;

 private:
    Viewport _viewport;
    CameraPtr _camera;
    RenderStates _renderStates;
    Color _backgroundColor = Color::makeBlack();
    std::vector<RenderablePtr> _renderables;
};

typedef std::shared_ptr<Renderer> RendererPtr;

} }  // namespace jet::viz

#endif  // INCLUDE_JET_VIZ_RENDERER_H_
