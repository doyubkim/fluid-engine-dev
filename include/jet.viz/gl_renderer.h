// Copyright (c) 2018 Doyub Kim
//
// I am making my contributions/submissions to this project solely in my
// personal capacity and am not conveying any rights to any intellectual
// property of any third parties.

#ifndef INCLUDE_JET_VIZ_GL_RENDERER_H_
#define INCLUDE_JET_VIZ_GL_RENDERER_H_

#ifdef JET_USE_GL

#include <jet.viz/renderer.h>

namespace jet {
namespace viz {

//! OpenGL renderer.
class GLRenderer final : public Renderer {
 public:
    //! Default constructor.
    GLRenderer();

    //! Destructor.
    virtual ~GLRenderer();

    //!
    //! Creates a vertex buffer with given parameters.
    //!
    //! \param shader Shader object for the buffer.
    //! \param vertices Vertex data.
    //! \param numberOfPoints Number of vertices.
    //! \return New vertex buffer.
    //!
    VertexBufferPtr createVertexBuffer(const ShaderPtr& shader,
                                       const float* vertices,
                                       size_t numberOfPoints) override;

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
    Texture2Ptr createTexture2(const ConstArrayAccessor2<ByteColor>& data) override;

    //!
    //! Creates a 2-D texture with 32-bit image and given parameters.
    //!
    //! \param data 32-bit texture image data.
    //! \param size Size of the data.
    //! \return New 2-D texture.
    //!
    Texture2Ptr createTexture2(const ConstArrayAccessor2<Color>& data) override;

    //!
    //! Creates a 3-D texture with 8-bit image and given parameters.
    //!
    //! \param data 8-bit texture image data.
    //! \param size Size of the data.
    //! \return New 3-D texture.
    //!
    Texture3Ptr createTexture3(const ConstArrayAccessor3<ByteColor>& data) override;

    //!
    //! Creates a 3-D texture with 32-bit image and given parameters.
    //!
    //! \param data 32-bit texture image data.
    //! \param size Size of the data.
    //! \return New 3-D texture.
    //!
    Texture3Ptr createTexture3(const ConstArrayAccessor3<Color>& data) override;

    //!
    //! Creates a shader object with given preset shader name.
    //!
    //! \param shaderName Preset shader name.
    //! \return
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

 protected:
    void onRenderBegin() override;
    void onRenderEnd() override;
    void onResize(const Viewport& viewport) override;
    void onSetRenderStates(const RenderStates& states) override;

 private:
    PrimitiveType _primitiveType;
};

typedef std::shared_ptr<GLRenderer> GLRendererPtr;

}  // namespace viz
}  // namespace jet

#endif  // JET_USE_GL

#endif  // INCLUDE_JET_VIZ_GL_RENDERER_H_
