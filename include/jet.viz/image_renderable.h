// Copyright (c) 2017 Doyub Kim
//
// I am making my contributions/submissions to this project solely in my
// personal capacity and am not conveying any rights to any intellectual
// property of any third parties.

#ifndef INCLUDE_JET_VIZ_IMAGE_RENDERABLE_H_
#define INCLUDE_JET_VIZ_IMAGE_RENDERABLE_H_

#include "image.h"
#include "renderable.h"
#include "shader.h"
#include "texture2.h"
#include "vertex_buffer.h"

namespace jet {

namespace viz {

//! Renderable for displaying a single image.
class ImageRenderable final : public Renderable {
 public:
    //! Constructs the renderable.
    explicit ImageRenderable(Renderer* renderer);

    //! Sets an image to be rendered.
    void setImage(const ByteImage& image);

    //! Sets the sampling mode for the image texture.
    void setTextureSamplingMode(const TextureSamplingMode& mode);

 private:
    Renderer* _renderer;
    ShaderPtr _shader;
    VertexBufferPtr _vertexBuffer;
    Texture2Ptr _texture;

    void render(Renderer* renderer) override;
};

//! Shared pointer type for ImageRenderable.
typedef std::shared_ptr<ImageRenderable> ImageRenderablePtr;

}  // namespace viz

}  // namespace jet

#endif  // INCLUDE_JET_VIZ_IMAGE_RENDERABLE_H_
