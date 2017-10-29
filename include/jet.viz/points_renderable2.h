// Copyright (c) 2017 Doyub Kim
//
// I am making my contributions/submissions to this project solely in my
// personal capacity and am not conveying any rights to any intellectual
// property of any third parties.

#ifndef INCLUDE_JET_VIZ_POINTS_RENDERABLE2_H_
#define INCLUDE_JET_VIZ_POINTS_RENDERABLE2_H_

#include <jet.viz/color.h>
#include <jet.viz/renderable.h>
#include <jet.viz/shader.h>
#include <jet.viz/vertex_buffer.h>

namespace jet {
namespace viz {

class PointsRenderable2 final : public Renderable {
 public:
    PointsRenderable2(Renderer* renderer);

    virtual ~PointsRenderable2();

    void setPositions(const Vector2F* positions, size_t numberOfParticles);

    void setPositionsAndColors(const Vector2F* positions, const Color* colors,
                               size_t numberOfParticles);

    float radius() const;

    void setRadius(float radius);

 protected:
    virtual void render(Renderer* renderer) override;

 private:
    Renderer* _renderer;
    ShaderPtr _shader;
    VertexBufferPtr _vertexBuffer;
    float _radius = 1.f;

    void updateVertexBuffer(const std::vector<VertexPosition3Color4>& vertices);
};

typedef std::shared_ptr<PointsRenderable2> PointsRenderable2Ptr;

}  // namespace viz
}  // namespace jet

#endif  // INCLUDE_JET_VIZ_POINTS_RENDERABLE2_H_
