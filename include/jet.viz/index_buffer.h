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

class IndexBuffer {
 public:
    IndexBuffer();
    virtual ~IndexBuffer();

    virtual void update(const std::uint32_t* indices) = 0;

    void clear();

    void resize(const VertexBufferPtr& vertexBuffer,
                const std::uint32_t* indices, std::size_t numberOfIndices);

    void bind(Renderer* renderer);

    void unbind(Renderer* renderer);

    std::size_t numberOfIndices() const;

 protected:
    virtual void onClear() = 0;

    virtual void onResize(const VertexBufferPtr& vertexBuffer,
                          const std::uint32_t* indices,
                          std::size_t numberOfIndices) = 0;

    virtual void onBind(Renderer* renderer) = 0;

    virtual void onUnbind(Renderer* renderer) = 0;

 private:
    std::size_t _numberOfIndices = 0;
};

typedef std::shared_ptr<IndexBuffer> IndexBufferPtr;

}  // namespace viz
}  // namespace jet

#endif  // INCLUDE_JET_VIZ_INDEX_BUFFER_H_
