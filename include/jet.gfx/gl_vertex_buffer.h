// Copyright (c) 2018 Doyub Kim
//
// I am making my contributions/submissions to this project solely in my
// personal capacity and am not conveying any rights to any intellectual
// property of any third parties.

#ifndef INCLUDE_JET_GFX_GL_VERTEX_BUFFER_H_
#define INCLUDE_JET_GFX_GL_VERTEX_BUFFER_H_

#ifdef JET_USE_GL

#include <jet.gfx/gl_common.h>
#include <jet.gfx/vertex_buffer.h>

#ifdef JET_USE_CUDA
#include <cuda_runtime.h>
#endif  // JET_USE_CUDA

#include <map>
#include <string>

namespace jet {
namespace gfx {

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
    void update(const float* vertexData) override;

#ifdef JET_USE_CUDA
    //!
    //! Updates the buffer with given CUDA vertex array.
    //!
    //! \param vertices Vertex array in CUDA device memory.
    //!
    void updateWithCuda(const float* vertexData) override;

    //! Maps vertex buffer to CUDA-accessible memory pointer.
    void* cudaMapResources() override;

    //! Unmaps vertex buffer from CUDA-accessible memory pointer.
    void cudaUnmapResources() override;
#endif  // JET_USE_CUDA

    //! Returns OpenGL vertex array ID.
    GLuint vertexArrayId() const;

    //! Returns OpenGL vertex buffer object ID.
    GLuint vertexBufferObjectId() const;

 private:
    GLuint _vertexArrayId = 0;
    GLuint _vertexBufferId = 0;
    GLuint _indexBufferId = 0;

    std::map<std::string, GLuint> _attributes;

#ifdef JET_USE_CUDA
    cudaGraphicsResource* _resource = nullptr;
#endif

    void onClear() override;

    void onResize(const ShaderPtr& shader, const float* vertexData,
                  size_t numberOfVertices) override;

    void onBind(Renderer* renderer) override;

    void onUnbind(Renderer* renderer) override;
};

//! Shared pointer type for GLVertexBuffer.
using GLVertexBufferPtr = std::shared_ptr<GLVertexBuffer>;

}  // namespace gfx
}  // namespace jet

#endif  // JET_USE_GL

#endif  // INCLUDE_JET_GFX_GL_VERTEX_BUFFER_H_
