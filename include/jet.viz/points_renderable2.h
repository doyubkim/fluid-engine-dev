// Copyright (c) 2018 Doyub Kim
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
#include <jet/array_view1.h>

namespace jet {
namespace viz {

//! 2-D points renderable.
class PointsRenderable2 final : public Renderable {
 public:
    //! Constructs a renderable.
    PointsRenderable2(Renderer* renderer);

    //! Destructor.
    virtual ~PointsRenderable2();

    //! Sets the positions of the points.
    void setPositions(const Vector2F* positions, size_t numberOfVertices);

    //! Sets the positions of the points.
    void setPositions(ConstArrayView1<Vector2F> positions);

    //! Sets the positions and colors of the points.
    void setPositionsAndColors(ConstArrayView1<Vector2F> positions,
                               ConstArrayView1<Color> colors);

    //! Sets the positions and colors of the points.
    void setPositionsAndColors(const Vector2F* positions, const Color* colors,
                               size_t numberOfVertices);

    //! Sets the positions and colors of the points.
    void setPositionsAndColors(const VertexPosition3Color4* vertices,
                               size_t numberOfVertices);

    //! Sets the positions and colors of the points.
    void setPositionsAndColors(ConstArrayView1<VertexPosition3Color4> vertices);

    //! Returns vertex buffer object.
    VertexBuffer* vertexBuffer() const;

    //! Returns radius of the points.
    float radius() const;

    //! Sets the radius of the points.
    void setRadius(float radius);

 protected:
    //! Renders points in 2-D space.
    virtual void render(Renderer* renderer) override;

 private:
    Renderer* _renderer;
    ShaderPtr _shader;
    VertexBufferPtr _vertexBuffer;
    float _radius = 1.f;

    void updateVertexBuffer(const std::vector<VertexPosition3Color4>& vertices);
};

//! Shared pointer type for PointsRenderable2.
typedef std::shared_ptr<PointsRenderable2> PointsRenderable2Ptr;

}  // namespace viz
}  // namespace jet

#endif  // INCLUDE_JET_VIZ_POINTS_RENDERABLE2_H_
