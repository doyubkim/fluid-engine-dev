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

#ifdef JET_USE_GL

#include <pch.h>

#include <jet.viz/gl_framebuffer_object.h>
#include <jet/logging.h>

using namespace jet;
using namespace viz;

GLFramebufferObject::GLFramebufferObject()
    : _fboId(generateFboId()), _savedFboId(0) {
    // Bind this FBO so that it actually gets created now
    guardedBind();
    guardedUnbind();
}

GLFramebufferObject::~GLFramebufferObject() {
    glDeleteFramebuffers(1, &_fboId);
}

void GLFramebufferObject::bind() { glBindFramebuffer(GL_FRAMEBUFFER, _fboId); }

void GLFramebufferObject::disable() { glBindFramebuffer(GL_FRAMEBUFFER, 0); }

void GLFramebufferObject::attachTexture(GLenum texTarget, GLuint texId,
                                        GLenum attachment, int mipLevel,
                                        int zSlice) {
    guardedBind();

#ifdef JET_DEBUG_MODE
    if (attachedId(attachment) != texId) {
#endif

        framebufferTextureND(attachment, texTarget, texId, mipLevel, zSlice);

#ifdef JET_DEBUG_MODE
    } else {
        JET_DEBUG << "Redundant bind of texture (id = " << texId << ").";
    }
#endif

    guardedUnbind();
}

void GLFramebufferObject::attachTextures(
    size_t numTextures, const GLenum* texTarget, const GLuint* texId,
    const GLenum* attachment, const int* mipLevel, const int* zSlice) {
    for (size_t i = 0; i < numTextures; ++i) {
        attachTexture(
            texTarget[i], texId[i],
            attachment ? attachment[i] : (GL_COLOR_ATTACHMENT0 + (GLenum)i),
            mipLevel ? mipLevel[i] : 0, zSlice ? zSlice[i] : 0);
    }
}

void GLFramebufferObject::attachRenderBuffer(GLuint buffId, GLenum attachment) {
    guardedBind();

    if (attachedId(attachment) != buffId) {
        glFramebufferRenderbuffer(GL_FRAMEBUFFER, attachment, GL_RENDERBUFFER,
                                  buffId);
    } else {
        JET_DEBUG << "Redundant bind of GLRenderBuffer (id = " << buffId << ")";
    }

    guardedUnbind();
}

void GLFramebufferObject::attachRenderBuffers(int numBuffers,
                                              const GLuint* buffId,
                                              const GLenum* attachment) {
    for (int i = 0; i < numBuffers; ++i) {
        attachRenderBuffer(buffId[i], attachment
                                          ? attachment[i]
                                          : (GL_COLOR_ATTACHMENT0 + (GLenum)i));
    }
}

void GLFramebufferObject::unattach(GLenum attachment) {
    guardedBind();
    GLenum type = attachedType(attachment);

    switch (type) {
        case GL_NONE:

        case GL_RENDERBUFFER:
            attachRenderBuffer(0, attachment);
            break;
        case GL_TEXTURE:
            attachTexture(GL_TEXTURE_2D, 0, attachment);
            break;
        default:
            JET_DEBUG << "Unknown attached resource type";
    }
    guardedUnbind();
}

void GLFramebufferObject::unattachAll() {
    size_t numAttachments = maxNumberOfColorAttachments();
    for (size_t i = 0; i < numAttachments; ++i) {
        unattach(GL_COLOR_ATTACHMENT0 + (GLenum)i);
    }
}

size_t GLFramebufferObject::maxNumberOfColorAttachments() {
    GLint maxAttach = 0;
    glGetIntegerv(GL_MAX_COLOR_ATTACHMENTS, &maxAttach);
    return (size_t)maxAttach;
}

GLuint GLFramebufferObject::generateFboId() {
    GLuint id = 0;
    glGenFramebuffers(1, &id);
    return id;
}

void GLFramebufferObject::guardedBind() {
    // Only binds if _fboId is different than the currently bound FBO
    glGetIntegerv(GL_FRAMEBUFFER_BINDING, &_savedFboId);
    if (_fboId != (GLuint)_savedFboId) {
        glBindFramebuffer(GL_FRAMEBUFFER, _fboId);
    }
}

void GLFramebufferObject::guardedUnbind() {
    // Returns FBO binding to the previously enabled FBO
    if (_fboId != (GLuint)_savedFboId) {
        glBindFramebuffer(GL_FRAMEBUFFER, (GLuint)_savedFboId);
    }
}

void GLFramebufferObject::framebufferTextureND(GLenum attachment,
                                               GLenum texTarget, GLuint texId,
                                               int mipLevel, int zSlice) {
    if (texTarget == GL_TEXTURE_1D) {
        glFramebufferTexture1D(GL_FRAMEBUFFER, attachment, GL_TEXTURE_1D, texId,
                               mipLevel);
    } else if (texTarget == GL_TEXTURE_3D) {
        glFramebufferTexture3D(GL_FRAMEBUFFER, attachment, GL_TEXTURE_3D, texId,
                               mipLevel, zSlice);
    } else {
        // Default is GL_TEXTURE_2D, GL_TEXTURE_RECTANGLE_ARB, or cube faces
        glFramebufferTexture2D(GL_FRAMEBUFFER, attachment, texTarget, texId,
                               mipLevel);
    }
}

bool GLFramebufferObject::isValid() {
    guardedBind();

    bool isOK = false;

    GLenum status;
    status = glCheckFramebufferStatus(GL_FRAMEBUFFER);
    switch (status) {
        case GL_FRAMEBUFFER_COMPLETE:  // Everything's OK
            isOK = true;
            break;
        case GL_FRAMEBUFFER_INCOMPLETE_ATTACHMENT:
            JET_DEBUG << "GL_FRAMEBUFFER_INCOMPLETE_ATTACHMENT";
            isOK = false;
            break;
        case GL_FRAMEBUFFER_INCOMPLETE_MISSING_ATTACHMENT:
            JET_DEBUG << "GL_FRAMEBUFFER_INCOMPLETE_MISSING_ATTACHMENT";
            isOK = false;
            break;
        case GL_FRAMEBUFFER_INCOMPLETE_DRAW_BUFFER:
            JET_DEBUG << "GL_FRAMEBUFFER_INCOMPLETE_DRAW_BUFFER";
            isOK = false;
            break;
        case GL_FRAMEBUFFER_INCOMPLETE_READ_BUFFER:
            JET_DEBUG << "GL_FRAMEBUFFER_INCOMPLETE_READ_BUFFER";
            isOK = false;
            break;
        case GL_FRAMEBUFFER_UNSUPPORTED:
            JET_DEBUG << "GL_FRAMEBUFFER_UNSUPPORTED";
            isOK = false;
            break;
        case GL_FRAMEBUFFER_INCOMPLETE_MULTISAMPLE:
            JET_DEBUG << "GL_FRAMEBUFFER_INCOMPLETE_MULTISAMPLE";
            isOK = false;
            break;
        case GL_FRAMEBUFFER_INCOMPLETE_LAYER_TARGETS:
            JET_DEBUG << "GL_FRAMEBUFFER_INCOMPLETE_LAYER_TARGETS";
            isOK = false;
            break;
        default:
            JET_DEBUG << "Unknown ERROR";
            isOK = false;
    }

    guardedUnbind();
    return isOK;
}

GLenum GLFramebufferObject::attachedType(GLenum attachment) {
    // Returns GL_RENDERBUFFER or GL_TEXTURE
    guardedBind();
    GLint type = 0;
    glGetFramebufferAttachmentParameteriv(GL_FRAMEBUFFER, attachment,
                                          GL_FRAMEBUFFER_ATTACHMENT_OBJECT_TYPE,
                                          &type);
    guardedUnbind();
    return GLenum(type);
}

GLuint GLFramebufferObject::attachedId(GLenum attachment) {
    guardedBind();
    GLint id = 0;
    glGetFramebufferAttachmentParameteriv(
        GL_FRAMEBUFFER, attachment, GL_FRAMEBUFFER_ATTACHMENT_OBJECT_NAME, &id);
    guardedUnbind();
    return GLuint(id);
}

GLint GLFramebufferObject::attachedMipLevel(GLenum attachment) {
    guardedBind();
    GLint level = 0;
    glGetFramebufferAttachmentParameteriv(
        GL_FRAMEBUFFER, attachment, GL_FRAMEBUFFER_ATTACHMENT_TEXTURE_LEVEL,
        &level);
    guardedUnbind();
    return level;
}

GLint GLFramebufferObject::attachedCubeFace(GLenum attachment) {
    guardedBind();
    GLint level = 0;
    glGetFramebufferAttachmentParameteriv(
        GL_FRAMEBUFFER, attachment,
        GL_FRAMEBUFFER_ATTACHMENT_TEXTURE_CUBE_MAP_FACE, &level);
    guardedUnbind();
    return level;
}

#endif  // JET_USE_GL
