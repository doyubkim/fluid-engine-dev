// Copyright (c) 2018 Doyub Kim
//
// I am making my contributions/submissions to this project solely in my
// personal capacity and am not conveying any rights to any intellectual
// property of any third parties.

#ifndef INCLUDE_JET_GFX_GLFW_WINDOW_H_
#define INCLUDE_JET_GFX_GLFW_WINDOW_H_

#ifdef JET_USE_GL

#include <jet.gfx/gl_renderer.h>
#include <jet.gfx/glfw_app.h>
#include <jet.gfx/window.h>
#include <jet/macros.h>

struct GLFWwindow;

namespace jet {
namespace gfx {

//!
//! \brief Helper class for GLFW-based window.
//!
//! \see GlfwApp
//!
class GlfwWindow final : public Window {
 public:
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

    //! Returns raw GLFW window object.
    GLFWwindow* glfwWindow() const;

 private:
    GLFWwindow* _window = nullptr;

    mutable bool _hasDisplayScalingFactorCache = false;
    mutable Vector2F _displayScalingFactorCache;

    MouseButtonType _pressedMouseButton = MouseButtonType::None;
    ModifierKey _lastModifierKey = ModifierKey::None;

    unsigned int _numRequestedRenderFrames = 0;

    int _width = 256;
    int _height = 256;
    float _pointerPosX = 0.0f;
    float _pointerPosY = 0.0f;
    float _pointerDeltaX = 0.0f;
    float _pointerDeltaY = 0.0f;

    int _swapInterval = 0;

    GlfwWindow(const std::string& title, int width, int height);

    void onRender();

    bool onWindowResized(int width, int height);

    bool onWindowMoved(int x, int y);

    bool onUpdate();

    bool onKey(int key, int scancode, int action, int mods);

    bool onPointerButton(int button, int action, int mods);

    bool onPointerMoved(float x, float y);

    bool onPointerEnter(bool entered);

    bool onMouseWheel(float deltaX, float deltaY);

    friend class GlfwApp;
};

//! Shared pointer type for GlfwWindow.
using GlfwWindowPtr = std::shared_ptr<GlfwWindow>;

}  // namespace gfx
}  // namespace jet

#endif  // JET_USE_GL

#endif  // INCLUDE_JET_GFX_GLFW_WINDOW_H_
