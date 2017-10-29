// Copyright (c) 2017 Doyub Kim
//
// I am making my contributions/submissions to this project solely in my
// personal capacity and am not conveying any rights to any intellectual
// property of any third parties.

#ifndef INCLUDE_JET_VIZ_SHADER_H_
#define INCLUDE_JET_VIZ_SHADER_H_

#include <jet.viz/render_parameters.h>
#include <jet.viz/vertex.h>
#include <jet/matrix4x4.h>

#include <map>
#include <memory>
#include <string>

namespace jet {
namespace viz {

class Renderer;

class Shader {
 public:
    Shader(const RenderParameters& userRenderParams);
    virtual ~Shader();

    virtual void clear() = 0;

    virtual void load(const VertexFormat& vertexFormat,
                      const std::string& vertexShaderSource,
                      const std::string& fragmentShaderSource) = 0;

    virtual void load(const VertexFormat& vertexFormat,
                      const std::string& vertexShaderSource,
                      const std::string& geometryShaderSource,
                      const std::string& fragmentShaderSource) = 0;

    void bind(Renderer* renderer);

    void unbind(Renderer* renderer);

    VertexFormat vertexFormat() const;

    const RenderParameters& defaultRenderParameters() const;

    const RenderParameters& userRenderParameters() const;

    void setModelViewProjectionMatrix(const Matrix4x4F& modelViewProjection);

    void setViewWidth(float viewWidth);

    void setViewHeight(float viewHeight);

    void setUserRenderParameter(const std::string& name, int32_t newValue);

    void setUserRenderParameter(const std::string& name, uint32_t newValue);

    void setUserRenderParameter(const std::string& name, float newValue);

    void setUserRenderParameter(const std::string& name,
                                const Vector2F& newValue);

    void setUserRenderParameter(const std::string& name,
                                const Vector3F& newValue);

    void setUserRenderParameter(const std::string& name,
                                const Vector4F& newValue);

    void setUserRenderParameter(const std::string& name,
                                const Matrix4x4F& newValue);

 protected:
    VertexFormat _vertexFormat = VertexFormat::Position3;

    virtual void onBind(Renderer* renderer) = 0;

    virtual void onUnbind(Renderer* renderer) = 0;

 private:
    RenderParameters _defaultRenderParams;
    RenderParameters _userRenderParams;
};

typedef std::shared_ptr<Shader> ShaderPtr;

}  // namespace viz
}  // namespace jet

#endif  // INCLUDE_JET_VIZ_SHADER_H_
