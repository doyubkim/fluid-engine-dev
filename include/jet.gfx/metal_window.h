// Copyright (c) 2018 Doyub Kim
//
// I am making my contributions/submissions to this project solely in my
// personal capacity and am not conveying any rights to any intellectual
// property of any third parties.

#ifndef INCLUDE_JET_GFX_METAL_WINDOW_H_
#define INCLUDE_JET_GFX_METAL_WINDOW_H_

#include <jet/macros.h>

#ifdef JET_MACOSX

#include <jet.gfx/event.h>
#include <jet.gfx/input_events.h>
#include <jet.gfx/metal_app.h>
#include <jet.gfx/metal_renderer.h>
#include <jet.gfx/window.h>

namespace jet {
namespace gfx {

class MetalView;
class MetalWindowEventHandler;

//!
//! \brief Helper class for Metal-based window.
//!
//! \see MetalApp
//!
class MetalWindow final : public Window {
 public:
    ~MetalWindow();

    //! Returns the framebuffer size.
    //! Note that the framebuffer size can be different from the window size,
    //! especially on a Retina display (2x the window size).
    Vector2UZ framebufferSize() const override;

    //! Returns the window size.
    Vector2UZ windowSize() const override;

    //! Request to render given number of frames to the renderer.
    void requestRender(unsigned int numFrames) override;

    //! Sets swap interval.
    void setSwapInterval(int interval) override;

    MetalView* view() const;

 private:
    MetalView* _view = nullptr;

    MetalWindow(const std::string& title, int width, int height);

    void onRender();

    friend class MetalApp;
    friend class MetalWindowEventHandler;
};

//! Shared pointer type for MetalWindow
using MetalWindowPtr = std::shared_ptr<MetalWindow>;

}  // namespace gfx
}  // namespace jet

#endif  // JET_MACOSX

#endif  // INCLUDE_JET_GFX_METAL_WINDOW_H_
