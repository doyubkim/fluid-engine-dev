// Copyright (c) 2017 Doyub Kim
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

class GLRenderer final : public Renderer {
 public:
    GLRenderer();
    virtual ~GLRenderer();

    virtual VertexBufferPtr createVertexBuffer(
        const ShaderPtr& shader, const float* vertices,
        std::size_t numberOfPoints) override;

    virtual IndexBufferPtr createIndexBuffer(
        const VertexBufferPtr& vertexBuffer, const std::uint32_t* indices,
        std::size_t numberOfIndices) override;

    virtual Texture2Ptr createTexture2(const std::uint8_t* const data,
                                       const Size2& size) override;

    virtual Texture2Ptr createTexture2(const float* const data,
                                       const Size2& size) override;

    virtual Texture3Ptr createTexture3(const float* const data,
                                       const Size3& size) override;

    virtual ShaderPtr createPresetShader(
        const std::string& shaderName) const override;

    virtual void setPrimitiveType(PrimitiveType type) override;

    virtual void draw(std::size_t numberOfVertices) override;

    virtual void drawIndexed(std::size_t numberOfIndices) override;

 protected:
    virtual void onRenderBegin() override;
    virtual void onRenderEnd() override;
    virtual void onResize(const Viewport& viewport) override;
    virtual void onSetRenderStates(const RenderStates& states) override;

 private:
    PrimitiveType _primitiveType;
};

typedef std::shared_ptr<GLRenderer> GLRendererPtr;

}  // namespace viz
}  // namespace jet

#endif  // JET_USE_GL

#endif  // INCLUDE_JET_VIZ_GL_RENDERER_H_
