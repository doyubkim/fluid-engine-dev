// Copyright (c) 2018 Doyub Kim
//
// I am making my contributions/submissions to this project solely in my
// personal capacity and am not conveying any rights to any intellectual
// property of any third parties.
//
// This source code is adopted from Aaron Lefohn, Robert Strzodka, and Adam
// Moerschell's framebuffer object class.
//
// Copyright (c) 2005,
// Aaron Lefohn	  (lefohn@cs.ucdavis.edu)
// Robert Strzodka (strzodka@stanford.edu)
// Adam Moerschell (atmoerschell@ucdavis.edu)
// All rights reserved.
//
// This software is licensed under the BSD open-source license. See
//         http://www.opensource.org/licenses/bsd-license.php for more detail.
//
// *************************************************************
// Redistribution and use in source and binary forms, with or
// without modification, are permitted provided that the following
//         conditions are met:
//
// Redistributions of source code must retain the above copyright notice,
// this list of conditions and the following disclaimer.
//
// Redistributions in binary form must reproduce the above copyright notice,
// this list of conditions and the following disclaimer in the documentation
// and/or other materials provided with the distribution.
//
// Neither the name of the University of Californa, Davis nor the names of
// the contributors may be used to endorse or promote products derived
//         from this software without specific prior written permission.
//
// THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS
// "AS IS" AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT
//         LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS
//         FOR A PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL
// THE COPYRIGHT OWNER OR CONTRIBUTORS BE LIABLE FOR ANY DIRECT,
//         INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
//         DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE
//         GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS
// INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY,
//         WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
//         (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF
// THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY
// OF SUCH DAMAGE.
//

#ifndef INCLUDE_JET_VIZ_GL_FRAMEBUFFER_OBJECT_H_
#define INCLUDE_JET_VIZ_GL_FRAMEBUFFER_OBJECT_H_

#ifdef JET_USE_GL

#include <jet.viz/gl_common.h>

#include <memory>

namespace jet {

namespace viz {

//!
//! \brief OpenGL framebuffer object.
//!
//! This class encapsulates the GLFramebufferObject (FBO) OpenGL spec. See the
//! official spec at:
//! http://oss.sgi.com/projects/ogl-sample/registry/EXT/framebuffer_object.txt
//! for details.
//!
//! A framebuffer object (FBO) is conceptually a structure containing pointers
//! to GPU memory. The memory pointed to is either an OpenGL texture or an
//! OpenGL RenderBuffer. FBOs can be used to render to one or more textures,
//! share depth buffers between multiple sets of color buffers/textures and are
//! a complete replacement for pbuffers.
//!
//! Performance Notes:
//! 1) It is more efficient (but not required) to call bind()
//! on an FBO before making multiple method calls. For example:
//!
//! \code{.cpp}
//! GLFramebufferObject fbo;
//! fbo.bind();
//! fbo.attachTexture(GL_TEXTURE_2D, texId0, GL_COLOR_ATTACHMENT0_EXT);
//! fbo.attachTexture(GL_TEXTURE_2D, texId1, GL_COLOR_ATTACHMENT1_EXT);
//! fbo.isValid();
//! \endcode
//!
//! To provide a complete encapsulation, the following usage
//! pattern works correctly but is less efficient:
//!
//! \code{.cpp}
//! GLFramebufferObject fbo;
//! // NOTE : No bind() call
//! fbo.attachTexture(GL_TEXTURE_2D, texId0, GL_COLOR_ATTACHMENT0_EXT);
//! fbo.attachTexture(GL_TEXTURE_2D, texId1, GL_COLOR_ATTACHMENT1_EXT);
//! fbo.isValid();
//! \endcode
//!
//! The first usage pattern binds the FBO only once, whereas
//! the second usage binds/unbinds the FBO for each method call.
//!
//! 2) Use GLFramebufferObject::disable() sparingly. We have intentionally left
//! out an "unbind()" method because it is largely unnecessary and encourages
//! rendundant Bind/Unbind coding. Binding an FBO is usually much faster than
//! enabling/disabling a pbuffer, but is still a costly operation. When
//! switching between multiple FBOs and a visible OpenGL framebuffer, the
//! following usage pattern is recommended:
//!
//! \code{.cpp}
//! GLFramebufferObject fbo1, fbo2;
//! fbo1.bind();
//! ... Render ...
//! // NOTE : No Unbind/disable here...
//!
//! fbo2.bind();
//! ... Render ...
//!
//! // disable FBO rendering and return to visible window
//! // OpenGL framebuffer.
//! GLFramebufferObject::disable();
//! \endcode
//!
class GLFramebufferObject final {
 public:
    //! Default constructor.
    GLFramebufferObject();

    //! Destructor.
    virtual ~GLFramebufferObject();

    //! Bind this FBO as current render target
    void bind();

    //! Bind a texture to the \p attachment point of this FBO
    void attachTexture(GLenum texTarget, GLuint texId, GLenum attachment,
                       int mipLevel = 0, int zSlice = 0);

    //!
    //! \brief Bind an array of textures to multiple \p attachment points of
    //! this FBO.
    //!
    //! By default, the first 'numTextures' attachments are used, starting with
    //! GL_COLOR_ATTACHMENT0_EXT.
    //!
    void attachTextures(size_t numTextures, const GLenum* texTarget,
                        const GLuint* texId, const GLenum* attachment = nullptr,
                        const int* mipLevel = nullptr,
                        const int* zSlice = nullptr);

    //! Bind a render buffer to the \p attachment point of this FBO
    void attachRenderBuffer(GLuint buffId, GLenum attachment);

    //!
    //! \brief Bind an array of render buffers to corresponding \p attachment
    //! points of this FBO.
    //!
    //! By default, the first 'numBuffers' attachments are used, starting with
    //! GL_COLOR_ATTACHMENT0_EXT.
    //!
    void attachRenderBuffers(int numBuffers, const GLuint* buffId,
                             const GLenum* attachment = nullptr);

    //! Free any resource bound to the \p attachment point of this FBO.
    void unattach(GLenum attachment);

    //! Free any resources bound to any attachment points of this FBO.
    void unattachAll();

    //! Is this FBO currently a valid render target?
    bool isValid();

    //! Get the FBO ID
    GLuint fboId() { return _fboId; }

    //! Is attached type GL_RENDERBUFFER_EXT or GL_TEXTURE?
    GLenum attachedType(GLenum attachment);

    //! What is the Id of GLRenderBuffer/texture currently
    //! attached to \p attachment?
    GLuint attachedId(GLenum attachment);

    //! Which mipmap level is currently attached to \p attachment?
    GLint attachedMipLevel(GLenum attachment);

    //! Which cube face is currently attached to \p attachment?
    GLint attachedCubeFace(GLenum attachment);

    //! Return number of color attachments permitted
    static size_t maxNumberOfColorAttachments();

    //!
    //! \brief Disable all FBO rendering and return to traditional,
    //! windowing-system controlled framebuffer.
    //!
    //! This is NOT an "unbind" for this specific FBO, but rather disables all
    //! FBO rendering. This call is intentionally "static" and named "disable"
    //! instead of "Unbind" for this reason. The motivation for this strange
    //! semantic is performance. Providing "Unbind" would likely lead to a large
    //! number of unnecessary FBO enablings/disabling.
    //!
    static void disable();

 private:
    GLuint _fboId;
    GLint _savedFboId;

    void guardedBind();
    void guardedUnbind();
    void framebufferTextureND(GLenum attachment, GLenum texTarget, GLuint texId,
                              int mipLevel, int zSlice);
    static GLuint generateFboId();
};

//! Shared pointer type for GLFramebufferObject.
typedef std::shared_ptr<GLFramebufferObject> GLFramebufferObjectPtr;

}  // namespace viz

}  // namespace jet

#endif  // JET_USE_GL

#endif  // INCLUDE_JET_VIZ_GL_FRAMEBUFFER_OBJECT_H_
