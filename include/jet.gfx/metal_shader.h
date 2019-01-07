// Copyright (c) 2018 Doyub Kim
//
// I am making my contributions/submissions to this project solely in my
// personal capacity and am not conveying any rights to any intellectual
// property of any third parties.

#ifndef INCLUDE_JET_GFX_METAL_SHADER_H_
#define INCLUDE_JET_GFX_METAL_SHADER_H_

#include <jet/macros.h>

#ifdef JET_MACOSX

#include <jet.gfx/shader.h>

namespace jet {
namespace gfx {

class MetalPrivateDevice;
class MetalPrivateLibrary;
class MetalPrivateFunction;
class MetalRenderer;

//! Metal shader.
class MetalShader final : public Shader {
 public:
    //!
    //! Constructs a shader with user-given render parameters, vertex format,
    //! vertex shader, and fragment shader.
    //!
    //! \param name             Shader name.
    //! \param device           Metal device.
    //! \param userRenderParams User-given render parameters.
    //! \param vertexFormat     Vertex format of this shader.
    //! \param shaderSource     Shader in string.
    //!
    MetalShader(const std::string& name, const MetalPrivateDevice* device,
                const RenderParameters& userRenderParams,
                const VertexFormat& vertexFormat,
                const std::string& shaderSource);

    //! Destructor.
    virtual ~MetalShader();

    //! Returns Metal Library pointer.
    MetalPrivateLibrary* library() const;

    //! Returns Metal Function pointer for vertex function (shader).
    MetalPrivateFunction* vertexFunction() const;

    //! Returns Metal Function pointer for fragment function (shader).
    MetalPrivateFunction* fragmentFunction() const;

    //! Returns the name of the shader.
    const std::string& name() const;

 private:
    std::string _name;
    MetalPrivateLibrary* _library = nullptr;
    MetalPrivateFunction* _vertFunc = nullptr;
    MetalPrivateFunction* _fragFunc = nullptr;

    size_t _vertexUniformSize = 0;
    std::map<std::string, size_t> _vertexUniformLocations;

    void onBind(const Renderer* renderer) override;

    void onUnbind(const Renderer* renderer) override;

    void clear() override;

    void load(const MetalPrivateDevice* device,
              const VertexFormat& vertexFormat,
              const std::string& shaderSource);

    friend class MetalRenderer;
};

typedef std::shared_ptr<MetalShader> MetalShaderPtr;

}  // namespace gfx
}  // namespace jet

#endif  // JET_MACOSX

#endif  // INCLUDE_JET_GFX_METAL_SHADER_H_
