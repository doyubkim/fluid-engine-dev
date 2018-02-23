// Copyright (c) 2018 Doyub Kim
//
// I am making my contributions/submissions to this project solely in my
// personal capacity and am not conveying any rights to any intellectual
// property of any third parties.

#ifndef INCLUDE_JET_VIZ_SIMPLE_VOLUME_RENDERABLE_H_
#define INCLUDE_JET_VIZ_SIMPLE_VOLUME_RENDERABLE_H_

#include <jet.viz/color.h>
#include <jet.viz/index_buffer.h>
#include <jet.viz/renderable.h>
#include <jet.viz/shader.h>
#include <jet.viz/texture3.h>
#include <jet.viz/vertex_buffer.h>
#include <jet/array_accessor3.h>
#include <jet/size3.h>

namespace jet {
namespace viz {

//!
//! \brief Simple volume rendering implementation using billboarding.
//!
//! This volume renderer visualizes a given volume data by stacking up multiple
//! polygons with 3-D texture attached.
//!
//! This code is adopted from Ingemar Rask and Johannes Schmid's work:
//! https://graphics.ethz.ch/teaching/former/imagesynthesis_06/miniprojects/p3/
//!
class SimpleVolumeRenderable final : public Renderable {
 public:
    //! Constructs the renderable.
    explicit SimpleVolumeRenderable(Renderer* renderer);

    //! Destructor.
    virtual ~SimpleVolumeRenderable();

    //! Sets the volume data to be rendered.
    void setVolume(const ConstArrayAccessor3<Color>& data);

    //! Returns the brightness multiplier of the volume rendering.
    float brightness() const;

    //! Sets the brightness multiplier of the volume rendering.
    void setBrightness(float newBrightness);

    //! Returns the density multiplier of the volume rendering.
    float density() const;

    //! Sets the density multiplier of the volume rendering.
    void setDensity(float newDensity);

    //! Returns the step-size between textured polygons.
    float stepSize() const;

    //! Sets the step-size between textured polygons.
    void setStepSize(float newStepSize);

    //! Makes a request to the renderable to update the vertex buffer.
    void requestUpdateVertexBuffer();

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

    bool _updateVertexBufferRequested = true;

    void render(Renderer* renderer) override;

    void updateVertexBuffer(Renderer* renderer);
};

typedef std::shared_ptr<SimpleVolumeRenderable> SimpleVolumeRenderablePtr;

}  // namespace viz
}  // namespace jet

#endif  // INCLUDE_JET_VIZ_SIMPLE_VOLUME_RENDERABLE_H_
