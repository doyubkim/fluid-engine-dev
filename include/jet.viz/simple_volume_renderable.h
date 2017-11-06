// Copyright (c) 2017 Doyub Kim
//
// I am making my contributions/submissions to this project solely in my
// personal capacity and am not conveying any rights to any intellectual
// property of any third parties.

#ifndef INCLUDE_JET_VIZ_SIMPLE_VOLUME_RENDERABLE_H_
#define INCLUDE_JET_VIZ_SIMPLE_VOLUME_RENDERABLE_H_

#include <jet/size3.h>
#include <jet.viz/color.h>
#include <jet.viz/index_buffer.h>
#include <jet.viz/renderable.h>
#include <jet.viz/shader.h>
#include <jet.viz/texture3.h>
#include <jet.viz/vertex_buffer.h>

namespace jet { namespace viz {

class SimpleVolumeRenderable final : public Renderable {
 public:
    SimpleVolumeRenderable(Renderer* renderer);

    virtual ~SimpleVolumeRenderable();

    void setVolume(const Color* data, const Size3& size);

    float brightness() const;

    void setBrightness(float newBrightness);

    float density() const;

    void setDensity(float newDensity);

    float stepSize() const;

    void setStepSize(float newStepSize);

 protected:
    void render(Renderer* renderer) override;

 private:
    Renderer* _renderer;
    ShaderPtr _shader;
    VertexBufferPtr _vertexBuffer;
    std::vector<IndexBufferPtr> _indexBuffers;
    Texture3Ptr _texture;

    float _brightness = 1.f;
    float _density = 1.f;
    float _stepSize = 0.01f;

    Vector3D _prevCameraLookAtDir;
    Vector3D _prevCameraOrigin;

    void updateVertexBuffer(Renderer* renderer);
};

typedef std::shared_ptr<SimpleVolumeRenderable> SimpleVolumeRenderablePtr;

} }  // namespace jet::viz

#endif  // INCLUDE_JET_VIZ_SIMPLE_VOLUME_RENDERABLE_H_
