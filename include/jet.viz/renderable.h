// Copyright (c) 2017 Doyub Kim
//
// I am making my contributions/submissions to this project solely in my
// personal capacity and am not conveying any rights to any intellectual
// property of any third parties.

#ifndef INCLUDE_JET_VIZ_RENDERABLE_H_
#define INCLUDE_JET_VIZ_RENDERABLE_H_

#include <memory>

namespace jet { namespace viz {

class Renderer;

class Renderable {
 public:
    Renderable();
    virtual ~Renderable();

 protected:
    friend class Renderer;

    virtual void render(Renderer* renderer) = 0;
};

typedef std::shared_ptr<Renderable> RenderablePtr;

} }  // namespace jet::viz

#endif  // INCLUDE_JET_VIZ_RENDERABLE_H_
