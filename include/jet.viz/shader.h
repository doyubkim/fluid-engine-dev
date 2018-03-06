// Copyright (c) 2018 Doyub Kim
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

//! Abstract based class for shader object.
class Shader {
 public:
    //!
    //! Constructs a shader with user-given render parameters.
    //!
    //! \param userRenderParams Render parameters.
    //!
    Shader(const RenderParameters& userRenderParams);

    //! Default destructor.
    virtual ~Shader();

    //! Clears the contents.
    virtual void clear() = 0;

    //!
    //! Loads vertex and fragment shaders.
    //!
    //! \param vertexFormat Vertex format.
    //! \param vertexShaderSource Vertex shader in string.
    //! \param fragmentShaderSource Fragment shader in string.
    //!
    virtual void load(const VertexFormat& vertexFormat,
                      const std::string& vertexShaderSource,
                      const std::string& fragmentShaderSource) = 0;

    //!
    //! Loads vertex, geometry, and fragment shaders.
    //!
    //! \param vertexFormat Vertex format.
    //! \param vertexShaderSource Vertex shader in string.
    //! \param geometryShader Geometry shader in string.
    //! \param fragmentShaderSource Fragment shader in string.
    //!
    virtual void load(const VertexFormat& vertexFormat,
                      const std::string& vertexShaderSource,
                      const std::string& geometryShaderSource,
                      const std::string& fragmentShaderSource) = 0;

    //! Binds the shader to given \p renderer.
    void bind(Renderer* renderer);

    //! Unbinds the shader from given \p renderer.
    void unbind(Renderer* renderer);

    //! Returns input vertex format for this shader.
    VertexFormat vertexFormat() const;

    //!
    //! \brief Returns default render parameters.
    //!
    //! This function returns the default render parameters including
    //! ModelViewProjection, ViewWidth, and ViewHeight.
    //!
    //! \return default render parameters.
    //!
    const RenderParameters& defaultRenderParameters() const;

    //! Returns user-given render parameters.
    const RenderParameters& userRenderParameters() const;

    //! Sets model-view-projection matrix.
    void setModelViewProjectionMatrix(const Matrix4x4F& modelViewProjection);

    //! Sets the view width.
    void setViewWidth(float viewWidth);

    //! Sets the view height.
    void setViewHeight(float viewHeight);

    //! Sets the user-given render parameter with given name and value.
    void setUserRenderParameter(const std::string& name, int32_t newValue);

    //! Sets the user-given render parameter with given name and value.
    void setUserRenderParameter(const std::string& name, uint32_t newValue);

    //! Sets the user-given render parameter with given name and value.
    void setUserRenderParameter(const std::string& name, float newValue);

    //! Sets the user-given render parameter with given name and value.
    void setUserRenderParameter(const std::string& name,
                                const Vector2F& newValue);

    //! Sets the user-given render parameter with given name and value.
    void setUserRenderParameter(const std::string& name,
                                const Vector3F& newValue);

    //! Sets the user-given render parameter with given name and value.
    void setUserRenderParameter(const std::string& name,
                                const Vector4F& newValue);

    //! Sets the user-given render parameter with given name and value.
    void setUserRenderParameter(const std::string& name,
                                const Matrix4x4F& newValue);

 protected:
    //! Input vertex format for this shader.
    VertexFormat _vertexFormat = VertexFormat::Position3;

    //! Called when the shader is bound.
    virtual void onBind(Renderer* renderer) = 0;

    //! Called when the shader is unbound.
    virtual void onUnbind(Renderer* renderer) = 0;

 private:
    RenderParameters _defaultRenderParams;
    RenderParameters _userRenderParams;
};

//! Shared pointer type for Shader.
typedef std::shared_ptr<Shader> ShaderPtr;

}  // namespace viz
}  // namespace jet

#endif  // INCLUDE_JET_VIZ_SHADER_H_
