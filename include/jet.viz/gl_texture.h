// Copyright (c) 2017 Doyub Kim
//
// I am making my contributions/submissions to this project solely in my
// personal capacity and am not conveying any rights to any intellectual
// property of any third parties.

#ifndef INCLUDE_JET_VIZ_GL_TEXTURE_H_
#define INCLUDE_JET_VIZ_GL_TEXTURE_H_

#ifdef JET_USE_GL

#include <jet/size3.h>

namespace jet {
namespace viz {

//! OpenGL texture parameters.
struct GLTextureParameters final {
    //! GL_TEXTURE_MIN_FILTER
    int minFilter;

    //! GL_TEXTURE_MAG_FILTER
    int magFilter;

    //! GL_TEXTURE_WRAP_S
    int wrapS;

    //! GL_TEXTURE_WRAP_T
    int wrapT;

    //! GL_TEXTURE_WRAP_R
    int wrapR;

    //! Default constructor.
    GLTextureParameters();
};

//! OpenGL texture class.
class GLTexture {
 public:
    //!
    //! Constructs texture with given target.
    //!
    //! \param target OpenGL texture target.
    //!
    explicit GLTexture(unsigned int target);

    //! Destructor.
    virtual ~GLTexture();

    //! Returns OpenGL texture parameters.
    const GLTextureParameters& glTextureParameters() const;

    //! Sets OpenGL texture parameters.
    void setGLTextureParamters(const GLTextureParameters& params);

 protected:
    //! Clears OpenGL texture resource.
    void clearGLTexture();

    //! \brief Creates OpenGL texture resource.
    //! This function will allocated texture resource for previously set texture
    //! parameters and store the generated texture ID internally.
    void createGLTexture();

    //! Binds OpenGL texture to current context.
    void bindGLTexture(unsigned int slotId);

    //! Returns OpenGL texture ID.
    unsigned int glTextureId() const;

    //! Returns OpenGL texture target.
    unsigned int glTarget() const;

 private:
    GLTextureParameters _param;
    unsigned int _texId = 0;
    unsigned int _target;
};

}  // namespace viz
}  // namespace jet

#endif  // JET_USE_GL

#endif  // INCLUDE_JET_VIZ_GL_TEXTURE_H_
