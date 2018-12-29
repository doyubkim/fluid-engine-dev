// Copyright (c) 2018 Doyub Kim
//
// I am making my contributions/submissions to this project solely in my
// personal capacity and am not conveying any rights to any intellectual
// property of any third parties.

#ifndef INCLUDE_JET_GFX_GL_SHADER_H_
#define INCLUDE_JET_GFX_GL_SHADER_H_

#ifdef JET_USE_GL

#include <jet.gfx/shader.h>

namespace jet {
namespace gfx {

//! OpenGL shader.
class GLShader final : public Shader {
 public:
    //!
    //! Constructs a shader with user-given render parameters.
    //!
    //! \param userRenderParams User-given render parameters.
    //!
    GLShader(const RenderParameters& userRenderParams);

    //!
    //! Constructs a shader with user-given render parameters, vertex format,
    //! vertex shader, and fragment shader.
    //!
    //! \param userRenderParams     User-given render parameters.
    //! \param vertexFormat         Vertex format.
    //! \param vertexShaderSource   Vertex shader in string.
    //! \param fragmentShaderSource Fragment shader in string.
    //!
    GLShader(const RenderParameters& userRenderParams,
             const VertexFormat& vertexFormat,
             const std::string& vertexShaderSource,
             const std::string& fragmentShaderSource);

    //!
    //! Constructs a shader with user-given render parameters, vertex format,
    //! vertex shader, geometry shader, and fragment shader.
    //!
    //! \param userRenderParams     User-given render parameters.
    //! \param vertexFormat         Vertex format.
    //! \param vertexShaderSource   Vertex shader in string.
    //! \param geometryShader       Geometry shader in string.
    //! \param fragmentShaderSource Fragment shader in string.
    //!
    GLShader(const RenderParameters& userRenderParams,
             const VertexFormat& vertexFormat,
             const std::string& vertexShaderSource,
             const std::string& geometryShader,
             const std::string& fragmentShaderSource);

    //! Destructor.
    virtual ~GLShader();

    //! Clears the contents.
    void clear() override;

    //!
    //! Loads vertex and fragment shaders.
    //!
    //! \param vertexFormat         Vertex format.
    //! \param vertexShaderSource   Vertex shader in string.
    //! \param fragmentShaderSource Fragment shader in string.
    //!
    void load(const VertexFormat& vertexFormat,
              const std::string& vertexShaderSource,
              const std::string& fragmentShaderSource);

    //!
    //! Loads vertex, geometry, and fragment shaders.
    //!
    //! \param vertexFormat         Vertex format.
    //! \param vertexShaderSource   Vertex shader in string.
    //! \param geometryShader       Geometry shader in string.
    //! \param fragmentShaderSource Fragment shader in string.
    //!
    void load(const VertexFormat& vertexFormat, const std::string& vertexShader,
              const std::string& geometryShader,
              const std::string& fragmentShader);

    //! Returns OpenGL program object handle.
    unsigned int program() const;

 private:
    unsigned int _program = 0;

    void onBind(const Renderer* renderer) override;

    void onUnbind(const Renderer* renderer) override;
};

typedef std::shared_ptr<GLShader> GLShaderPtr;

}  // namespace gfx
}  // namespace jet

#endif  // JET_USE_GL

#endif  // INCLUDE_JET_GFX_GL_SHADER_H_
