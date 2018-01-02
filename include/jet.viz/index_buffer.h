// Copyright (c) 2017 Doyub Kim
//
// I am making my contributions/submissions to this project solely in my
// personal capacity and am not conveying any rights to any intellectual
// property of any third parties.

#ifndef INCLUDE_JET_VIZ_INDEX_BUFFER_H_
#define INCLUDE_JET_VIZ_INDEX_BUFFER_H_

#include <jet.viz/shader.h>
#include <jet.viz/vertex_buffer.h>

namespace jet {
namespace viz {

class Renderer;

//! Abstract base class for index buffers.
class IndexBuffer {
 public:
    //! Constructs an empty buffer.
    IndexBuffer();

    //! Destructor.
    virtual ~IndexBuffer();

    //!
    //! Updates the buffer with given indices array.
    //!
    //! \param indices Index array.
    //!
    virtual void update(const uint32_t* indices) = 0;

    //! Clears the buffer.
    void clear();

    //!
    //! \brief Resizes the buffer with given parameters.
    //!
    //! \param vertexBuffer The associated vertex buffer.
    //! \param indices Index array.
    //! \param numberOfIndices Number of indices.
    //!
    void resize(const VertexBufferPtr& vertexBuffer, const uint32_t* indices,
                size_t numberOfIndices);

    //!
    //! \brief Binds the buffer to the renderer.
    //!
    //! \param renderer The renderer in current render context.
    //!
    void bind(Renderer* renderer);

    //!
    //! \brief Unbinds the buffer to the renderer.
    //!
    //! \param renderer The renderer in current render context.
    //!
    void unbind(Renderer* renderer);

    //! Returns the number of indices.
    size_t numberOfIndices() const;

 protected:
    //! Called when clear() is invoked.
    virtual void onClear() = 0;

    //! Called when resize(...) is invoked.
    virtual void onResize(const VertexBufferPtr& vertexBuffer,
                          const uint32_t* indices, size_t numberOfIndices) = 0;

    //! Called when bind(...) is invoked.
    virtual void onBind(Renderer* renderer) = 0;

    //! Called when unbind(...) is invoked.
    virtual void onUnbind(Renderer* renderer) = 0;

 private:
    size_t _numberOfIndices = 0;
};

typedef std::shared_ptr<IndexBuffer> IndexBufferPtr;

}  // namespace viz
}  // namespace jet

#endif  // INCLUDE_JET_VIZ_INDEX_BUFFER_H_
