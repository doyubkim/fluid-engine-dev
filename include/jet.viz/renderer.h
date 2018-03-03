// Copyright (c) 2018 Doyub Kim
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

#include <jet/array_accessor2.h>
#include <jet/array_accessor3.h>

#include <map>
#include <string>
#include <vector>

namespace jet {

namespace viz {

//!
//! \brief Abstract base class for renderer.
//!
//! This class defines the interface for the renderers. Actual renderers should
//! override purely virtual functions in order to implement its own rendering
//! backend (such as OpenGL, Metal, Vulkan, D3D, etc).
//!
class Renderer {
 public:
    //! Default constructor.
    Renderer();

    //! Default destructor.
    virtual ~Renderer();

    //!
    //! Creates a vertex buffer with given parameters.
    //!
    //! \param shader Shader object for the buffer.
    //! \param vertices Vertex data.
    //! \param numberOfPoints Number of vertices.
    //! \return New vertex buffer.
    //!
    virtual VertexBufferPtr createVertexBuffer(const ShaderPtr& shader,
                                               const float* vertices,
                                               size_t numberOfPoints) = 0;

    //!
    //! Creates an index buffer with given parameters.
    //!
    //! \param vertexBuffer Vertices for the index buffer.
    //! \param indices Index data.
    //! \param numberOfIndices Number of indices.
    //! \return New index buffer.
    //!
    virtual IndexBufferPtr createIndexBuffer(
        const VertexBufferPtr& vertexBuffer, const uint32_t* indices,
        size_t numberOfIndices) = 0;

    //!
    //! Creates a 2-D texture with 8-bit image and given parameters.
    //!
    //! \param data 8-bit texture image data.
    //! \param size Size of the data.
    //! \return New 2-D texture.
    //!
    virtual Texture2Ptr createTexture2(
        const ConstArrayAccessor2<ByteColor>& data) = 0;

    //!
    //! Creates a 2-D texture with 32-bit image and given parameters.
    //!
    //! \param data 32-bit texture image data.
    //! \param size Size of the data.
    //! \return New 2-D texture.
    //!
    virtual Texture2Ptr createTexture2(
        const ConstArrayAccessor2<Color>& data) = 0;

    //!
    //! Creates a 3-D texture with 8-bit image and given parameters.
    //!
    //! \param data 8-bit texture image data.
    //! \param size Size of the data.
    //! \return New 3-D texture.
    //!
    virtual Texture3Ptr createTexture3(
        const ConstArrayAccessor3<ByteColor>& data) = 0;

    //!
    //! Creates a 3-D texture with 32-bit image and given parameters.
    //!
    //! \param data 32-bit texture image data.
    //! \param size Size of the data.
    //! \return New 3-D texture.
    //!
    virtual Texture3Ptr createTexture3(
        const ConstArrayAccessor3<Color>& data) = 0;

    //!
    //! Creates a shader object with given preset shader name.
    //!
    //! \param shaderName Preset shader name.
    //! \return New shader.
    //!
    virtual ShaderPtr createPresetShader(
        const std::string& shaderName) const = 0;

    //!
    //! Sets current render primitive type state for drawing.
    //!
    //! \param type Primitive type.
    //!
    virtual void setPrimitiveType(PrimitiveType type) = 0;

    //!
    //! Draws currently bound object.
    //!
    //! \param numberOfVertices Number of vertices.
    //!
    virtual void draw(size_t numberOfVertices) = 0;

    //!
    //! Draws currently bound indexed object.
    //!
    //! \param numberOfIndices Number of indices.
    //!
    virtual void drawIndexed(size_t numberOfIndices) = 0;

    //! Renders a frame.
    void render();

    //! Resizes the viewport.
    void resize(const Viewport& viewport);

    //! Adds a renderable to the renderer.
    void addRenderable(const RenderablePtr& renderable);

    //! Clears all renderables.
    void clearRenderables();

    //! Binds a shader to the current rendering context.
    void bindShader(const ShaderPtr& shader);

    //! Unbinds a shader to the current rendering context.
    void unbindShader(const ShaderPtr& shader);

    //! Binds a vertex buffer to the current rendering context.
    void bindVertexBuffer(const VertexBufferPtr& vertexBuffer);

    //! Unbinds a vertex to the current rendering context.
    void unbindVertexBuffer(const VertexBufferPtr& vertexBuffer);

    //! Binds a index buffer to the current rendering context.
    void bindIndexBuffer(const IndexBufferPtr& indexBuffer);

    //! Unbinds a index buffer to the current rendering context.
    void unbindIndexBuffer(const IndexBufferPtr& indexBuffer);

    //! Binds a 2-D texture to the current rendering context.
    void bindTexture(const Texture2Ptr& texture, unsigned int slotId);

    //! Binds a 3-D texture to the current rendering context.
    void bindTexture(const Texture3Ptr& texture, unsigned int slotId);

    //! Returns current viewport.
    const Viewport& viewport() const;

    //! Returns current camera.
    const CameraPtr& camera() const;

    //! Sets camera.
    void setCamera(const CameraPtr& camera);

    //! Returns current render states.
    const RenderStates& renderStates() const;

    //! Sets current render states.
    void setRenderStates(const RenderStates& states);

    //! Returns current background color.
    const Color& backgroundColor() const;

    //! Sets current background color.
    void setBackgroundColor(const Color& color);

 protected:
    //! Called when rendering a frame begins.
    virtual void onRenderBegin() = 0;

    //! Called when rendering a frame ended.
    virtual void onRenderEnd() = 0;

    //! Called when the view has resized.
    virtual void onResize(const Viewport& viewport) = 0;

    //! Called when the render states has changed.
    virtual void onSetRenderStates(const RenderStates& states) = 0;

 private:
    Viewport _viewport;
    CameraPtr _camera;
    RenderStates _renderStates;
    Color _backgroundColor = Color::makeBlack();
    std::vector<RenderablePtr> _renderables;
};

//!
typedef std::shared_ptr<Renderer> RendererPtr;

}  // namespace viz

}  // namespace jet

#endif  // INCLUDE_JET_VIZ_RENDERER_H_
