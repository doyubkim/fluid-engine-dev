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

// TODO: Make this a common parameter set
struct GLTextureParameters final {
    int minFilter;
    int magFilter;
    int wrapS;
    int wrapT;
    int wrapR;

    GLTextureParameters();
};

class GLTexture {
 public:
    GLTexture(unsigned int target);

    virtual ~GLTexture();

 protected:
    void clearGLTexture();

    void createGLTexture();

    void bindGLTexture(unsigned int slotId);

    const GLTextureParameters& glParameters() const;

    unsigned int glTextureId() const;

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
