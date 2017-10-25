// Copyright (c) 2017 Doyub Kim
//
// I am making my contributions/submissions to this project solely in my
// personal capacity and am not conveying any rights to any intellectual
// property of any third parties.

#ifndef INCLUDE_JET_VIZ_GL_SHADER_H_
#define INCLUDE_JET_VIZ_GL_SHADER_H_

#ifdef JET_USE_GL

#include <jet.viz/shader.h>

namespace jet {
namespace viz {

class GLShader final : public Shader {
 public:
    GLShader(const RenderParameters& userRenderParams);
    GLShader(const RenderParameters& userRenderParams,
             const VertexFormat& vertexFormat,
             const std::string& vertexShaderSource,
             const std::string& fragmentShaderSource);
    GLShader(const RenderParameters& userRenderParams,
             const VertexFormat& vertexFormat,
             const std::string& vertexShaderSource,
             const std::string& geometryShader,
             const std::string& fragmentShaderSource);

    virtual ~GLShader();

    virtual void clear() override;

    virtual void load(const VertexFormat& vertexFormat,
                      const std::string& vertexShaderSource,
                      const std::string& fragmentShaderSource) override;

    virtual void load(const VertexFormat& vertexFormat,
                      const std::string& vertexShader,
                      const std::string& geometryShader,
                      const std::string& fragmentShader) override;

    unsigned int program() const;

 protected:
    virtual void onBind(Renderer* renderer) override;

    virtual void onUnbind(Renderer* renderer) override;

 private:
    unsigned int _program = 0;
};

typedef std::shared_ptr<GLShader> GLShaderPtr;

}  // namespace viz
}  // namespace jet

#endif  // JET_USE_GL

#endif  // INCLUDE_JET_VIZ_GL_SHADER_H_
