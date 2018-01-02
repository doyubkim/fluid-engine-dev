// Copyright (c) 2017 Doyub Kim
//
// I am making my contributions/submissions to this project solely in my
// personal capacity and am not conveying any rights to any intellectual
// property of any third parties.

#ifndef INCLUDE_JET_VIZ_GL_INDEX_BUFFER_H_
#define INCLUDE_JET_VIZ_GL_INDEX_BUFFER_H_

#ifdef JET_USE_GL

#include <jet.viz/index_buffer.h>

namespace jet {

namespace viz {

//!
//! \brief Dynamic index buffer implementation for OpenGL.
//!
//! This class implements OpenGL version of index buffer. This index buffer is
//! dynamic, meaning that the contents can be updated dynamically
//! (GL_DYNAMIC_DRAW).
//!
class GLIndexBuffer final : public IndexBuffer {
 public:
    //! Constructs an empty buffer.
    GLIndexBuffer();

    //! Releases OpenGL buffer object.
    virtual ~GLIndexBuffer();

    //!
    //! Updates the buffer with given indices array.
    //!
    //! \param indices Index array.
    //!
    void update(const uint32_t* indices) override;

 private:
    unsigned int _bufferId = 0;

    void onClear() override;

    void onResize(const VertexBufferPtr& vertexBuffer, const uint32_t* indices,
                  size_t numberOfIndices) override;

    void onBind(Renderer* renderer) override;

    void onUnbind(Renderer* renderer) override;
};

//! Shared pointer type for GLIndexBuffer.
typedef std::shared_ptr<GLIndexBuffer> GLIndexBufferPtr;

}  // namespace viz

}  // namespace jet

#endif  // JET_USE_GL

#endif  // INCLUDE_JET_VIZ_GL_INDEX_BUFFER_H_
