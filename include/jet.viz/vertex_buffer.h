// Copyright (c) 2017 Doyub Kim
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

class VertexBuffer {
 public:
    VertexBuffer();
    virtual ~VertexBuffer();

    virtual void update(const float* vertices) = 0;

    void clear();

    void resize(const ShaderPtr& shader, const float* vertices,
                size_t numberOfVertices);

    void bind(Renderer* renderer);

    void unbind(Renderer* renderer);

    size_t numberOfVertices() const;

    VertexFormat vertexFormat() const;

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
