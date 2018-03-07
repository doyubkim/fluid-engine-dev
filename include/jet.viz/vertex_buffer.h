// Copyright (c) 2018 Doyub Kim
//
// I am making my contributions/submissions to this project solely in my
// personal capacity and am not conveying any rights to any intellectual
// property of any third parties.

#ifndef INCLUDE_JET_VIZ_VERTEX_BUFFER_H_
#define INCLUDE_JET_VIZ_VERTEX_BUFFER_H_

#include <jet.viz/primitive_types.h>
#include <jet.viz/shader.h>
#include <jet.viz/vertex.h>

namespace jet {

namespace viz {

class Renderer;

//! Abstract base class for vertex buffer types.
class VertexBuffer {
 public:
    //! Default constructor.
    VertexBuffer();

    //! Default destructor.
    virtual ~VertexBuffer();

    //!
    //! Updates the buffer with given vertex array.
    //!
    //! \param vertices Vertex array.
    //!
    virtual void update(const float* vertices) = 0;

#ifdef JET_USE_CUDA
    //!
    //! Updates the buffer with given CUDA vertex array.
    //!
    //! \param vertices Vertex array in CUDA device memory.
    //!
    virtual void updateWithCuda(const float* vertices);

    //! Maps vertex buffer to CUDA-accessible memory pointer.
    virtual void* cudaMapResources();

    //! Unmaps vertex buffer from CUDA-accessible memory pointer.
    virtual void cudaUnmapResources();
#endif

    //! Clears the contents.
    void clear();

    //! Resizes the buffer.
    void resize(const ShaderPtr& shader, const float* vertices,
                size_t numberOfVertices);

    //! Binds this buffer to the given \p renderer.
    void bind(Renderer* renderer);

    //! Unbinds this buffer from the given \p renderer.
    void unbind(Renderer* renderer);

    //! Returns the number of vertices.
    size_t numberOfVertices() const;

    //! Returns vertex format.
    VertexFormat vertexFormat() const;

    //! Returns shader object that is associated with this buffer.
    const ShaderPtr& shader() const;

 protected:
    virtual void onClear() = 0;

    virtual void onResize(const ShaderPtr& shader, const float* vertices,
                          size_t numberOfVertices) = 0;

    virtual void onBind(Renderer* renderer) = 0;

    virtual void onUnbind(Renderer* renderer) = 0;

 private:
    size_t _numberOfVertices = 0;
    VertexFormat _vertexFormat = VertexFormat::Position3;
    ShaderPtr _shader;
};

typedef std::shared_ptr<VertexBuffer> VertexBufferPtr;

}  // namespace viz

}  // namespace jet

#endif  // INCLUDE_JET_VIZ_VERTEX_BUFFER_H_
