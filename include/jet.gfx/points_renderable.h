// Copyright (c) 2018 Doyub Kim
//
// I am making my contributions/submissions to this project solely in my
// personal capacity and am not conveying any rights to any intellectual
// property of any third parties.

#ifndef INCLUDE_JET_GFX_POINTS_RENDERABLE_H_
#define INCLUDE_JET_GFX_POINTS_RENDERABLE_H_

#include <jet.gfx/renderable.h>
#include <jet.gfx/shader.h>
#include <jet.gfx/vertex_buffer.h>
#include <jet/array_view.h>
#include <jet/matrix.h>

#include <mutex>

namespace jet {
namespace gfx {

class PointsRenderable final : public Renderable {
 public:
    PointsRenderable(const ConstArrayView1<Vector3F>& positions,
                     const ConstArrayView1<Vector4F>& colors, float radius);

    void update(const ConstArrayView1<Vector3F>& positions,
                const ConstArrayView1<Vector4F>& colors, float radius);

 private:
    Array1<VertexPosition3Color4> _vertices;
    float _radius;

    std::mutex _dataMutex;

    VertexBufferPtr _vertexBuffer;
    ShaderPtr _shader;

    void onInitializeGpuResources(Renderer* renderer) override;

    void onRender(Renderer* renderer) override;
};

}  // namespace gfx
}  // namespace jet

#endif  // INCLUDE_JET_GFX_POINTS_RENDERABLE_H_
