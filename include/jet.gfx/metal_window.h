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

class MetalPrivateWindow;
class MetalPrivateView;
class MetalCustomViewEventHandler;

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

    //! Returns framebuffer / window size ratio.
    Vector2F displayScalingFactor() const override;

    //! Request to render given number of frames to the renderer.
    void requestRender(unsigned int numFrames) override;

    //! Sets swap interval.
    void setSwapInterval(int interval) override;

    MetalPrivateWindow* window() const;

    MetalPrivateView* view() const;

 private:
    MetalPrivateWindow* _window = nullptr;
    MetalPrivateView* _view = nullptr;

    unsigned int _numRequestedRenderFrames = 0;

    int _width = 256;
    int _height = 256;

    MetalWindow(const std::string& title, int width, int height);

    void onUpdateEnabled(bool enabled) override;

    void onRender();

    bool onWindowResized(int width, int height);

    bool onWindowMoved(int x, int y);

    bool onUpdate();

    bool onKeyDown(const KeyEvent& keyEvent);

    bool onKeyUp(const KeyEvent& keyEvent);

    bool onMouseDown(MouseButtonType button, ModifierKey mods, float x,
                     float y);

    bool onMouseUp(MouseButtonType button, ModifierKey mods, float x, float y);

    bool onMouseDragged(MouseButtonType button, ModifierKey mods, float x,
                        float y, float dx, float dy);

    bool onMouseHover(ModifierKey mods, float x, float y, float dx, float dy);

    bool onMouseScrollWheel(ModifierKey mods, float x, float y, float dx,
                            float dy);

    bool onMouseEntered(bool entered);

    friend class MetalApp;
    friend class MetalCustomViewEventHandler;
};

//! Shared pointer type for MetalWindow
using MetalWindowPtr = std::shared_ptr<MetalWindow>;

}  // namespace gfx
}  // namespace jet

#endif  // JET_MACOSX

#endif  // INCLUDE_JET_GFX_METAL_WINDOW_H_
