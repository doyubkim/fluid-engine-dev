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

namespace jet { namespace viz {

class ImageRenderable final : public Renderable {
 public:
    ImageRenderable(Renderer* renderer);

    void setImage(const ByteImage& image);

 protected:
    void render(Renderer* renderer) override;

 private:
    Renderer* _renderer;
    ShaderPtr _shader;
    VertexBufferPtr _vertexBuffer;
    Texture2Ptr _texture;
};

typedef std::shared_ptr<ImageRenderable> ImageRenderablePtr;

} }  // namespace jet::viz

#endif  // INCLUDE_JET_VIZ_IMAGE_RENDERABLE_H_
