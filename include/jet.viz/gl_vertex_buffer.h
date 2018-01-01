// Copyright (c) 2017 Doyub Kim
//
// I am making my contributions/submissions to this project solely in my
// personal capacity and am not conveying any rights to any intellectual
// property of any third parties.

#ifndef INCLUDE_JET_VIZ_GL_VERTEX_BUFFER_H_
#define INCLUDE_JET_VIZ_GL_VERTEX_BUFFER_H_

#ifdef JET_USE_GL

#include <jet.viz/vertex_buffer.h>

#include <map>
#include <string>

namespace jet {
namespace viz {

//!
//! \brief Dynamic vertex buffer implementation for OpenGL.
//!
//! This class implements OpenGL version of vertex buffer. This vertex
//! buffer is dynamic, meaning that the contents can be updated dynamically
//! (GL_DYNAMIC_DRAW).
//!
class GLVertexBuffer final : public VertexBuffer {
 public:
    //! Default constructor.
    GLVertexBuffer();

    //! Destructor.
    virtual ~GLVertexBuffer();

    //!
    //! Updates the buffer with given vertex array.
    //!
    //! \param vertices Vertex array.
    //!
    void update(const float* vertices) override;

    //! Returns OpenGL vertex array ID.
    unsigned int vertexArrayId() const;

 private:
    unsigned int _vertexArrayId = 0;
    unsigned int _vertexBufferId = 0;
    unsigned int _indexBufferId = 0;

    std::map<std::string, unsigned int> _attributes;

    void onClear() override;

    void onResize(const ShaderPtr& shader, const float* vertices,
                  size_t numberOfVertices) override;

    void onBind(Renderer* renderer) override;

    void onUnbind(Renderer* renderer) override;
};

//! Shared pointer type for GLVertexBuffer.
typedef std::shared_ptr<GLVertexBuffer> GLVertexBufferPtr;

}  // namespace viz
}  // namespace jet

#endif  // JET_USE_GL

#endif  // INCLUDE_JET_VIZ_GL_VERTEX_BUFFER_H_
