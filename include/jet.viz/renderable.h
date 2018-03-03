// Copyright (c) 2018 Doyub Kim
//
// I am making my contributions/submissions to this project solely in my
// personal capacity and am not conveying any rights to any intellectual
// property of any third parties.

#ifndef INCLUDE_JET_VIZ_RENDERABLE_H_
#define INCLUDE_JET_VIZ_RENDERABLE_H_

#include <memory>

namespace jet {

namespace viz {

class Renderer;

//! Abstract renderable representation.
class Renderable {
 public:
    //! Default constructor.
    Renderable();

    //! Default destructor.
    virtual ~Renderable();

 protected:
    friend class Renderer;

    //! Renders this renderable for given \p renderer context.
    virtual void render(Renderer* renderer) = 0;
};

//! Shared pointer type for Renderable.
typedef std::shared_ptr<Renderable> RenderablePtr;

}  // namespace viz

}  // namespace jet

#endif  // INCLUDE_JET_VIZ_RENDERABLE_H_
