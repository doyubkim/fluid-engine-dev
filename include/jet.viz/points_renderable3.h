// Copyright (c) 2018 Doyub Kim
//
// I am making my contributions/submissions to this project solely in my
// personal capacity and am not conveying any rights to any intellectual
// property of any third parties.

#ifndef INCLUDE_JET_VIZ_POINTS_RENDERABLE3_H_
#define INCLUDE_JET_VIZ_POINTS_RENDERABLE3_H_

#include <jet.viz/color.h>
#include <jet.viz/renderable.h>
#include <jet.viz/shader.h>
#include <jet.viz/vertex_buffer.h>
#include <jet/array_view1.h>

namespace jet {
namespace viz {

//! 3-D points renderable.
class PointsRenderable3 final : public Renderable {
 public:
    //! Constructs a renderable.
    PointsRenderable3(Renderer* renderer);

    //! Destructor.
    virtual ~PointsRenderable3();

    //! Sets the positions of the points.
    void setPositions(const Vector3F* positions, size_t numberOfVertices);

    //! Sets the positions of the points.
    void setPositions(ConstArrayView1<Vector3F> positions);

    //! Sets the positions and colors of the points.
    void setPositionsAndColors(const Vector3F* positions, const Color* colors,
                               size_t numberOfVertices);

    //! Sets the positions and colors of the points.
    void setPositionsAndColors(ConstArrayView1<Vector3F> positions,
                               ConstArrayView1<Color> colors);

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
    //! Renders points in 3-D space.
    virtual void render(Renderer* renderer) override;

 private:
    Renderer* _renderer;
    ShaderPtr _shader;
    VertexBufferPtr _vertexBuffer;
    float _radius = 1.f;
};

typedef std::shared_ptr<PointsRenderable3> PointsRenderable3Ptr;

}  // namespace viz
}  // namespace jet

#endif  // INCLUDE_JET_VIZ_POINTS_RENDERABLE3_H_
