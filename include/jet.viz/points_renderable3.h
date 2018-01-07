// Copyright (c) 2017 Doyub Kim
//
// I am making my contributions/submissions to this project solely in my
// personal capacity and am not conveying any rights to any intellectual
// property of any third parties.

#ifndef INCLUDE_JET_VIZ_POINTS_RENDERABLE3_H_
#define INCLUDE_JET_VIZ_POINTS_RENDERABLE3_H_

#include "color.h"
#include "renderable.h"
#include "shader.h"
#include "vertex_buffer.h"

#ifdef JET_USE_CUDA
#include <cuda_runtime.h>
#endif  // JET_USE_CUDA

namespace jet {
namespace viz {

class PointsRenderable3 final : public Renderable {
 public:
    PointsRenderable3(Renderer* renderer);

    virtual ~PointsRenderable3();

    void setPositions(const Vector3F* positions, size_t numberOfVertices);

    void setPositionsAndColors(const Vector3F* positions, const Color* colors,
                               size_t numberOfVertices);

    void setPositionsAndColors(const VertexPosition3Color4* deviceData,
                               size_t numberOfVertices);

    VertexBuffer* vertexBuffer() const;

    float radius() const;

    void setRadius(float radius);

 protected:
    virtual void render(Renderer* renderer) override;

 private:
    Renderer* _renderer;
    ShaderPtr _shader;
    VertexBufferPtr _vertexBuffer;
    float _radius = 1.f;

#ifdef JET_USE_CUDA
    //! CUDA graphics resource handle.
    cudaGraphicsResource* _resource = nullptr;
#endif
};

typedef std::shared_ptr<PointsRenderable3> PointsRenderable3Ptr;

}  // namespace viz
}  // namespace jet

#endif  // INCLUDE_JET_VIZ_POINTS_RENDERABLE3_H_
