// Copyright (c) 2018 Doyub Kim
//
// I am making my contributions/submissions to this project solely in my
// personal capacity and am not conveying any rights to any intellectual
// property of any third parties.

#ifndef INCLUDE_JET_GFX_METAL_APP_H_
#define INCLUDE_JET_GFX_METAL_APP_H_

#include <jet/macros.h>

#ifdef JET_MACOSX

#include <jet.gfx/event.h>

#include <memory>
#include <string>

namespace jet {
namespace gfx {

class MetalWindow;
using MetalWindowPtr = std::shared_ptr<MetalWindow>;

//!
//! \brief Helper class for macOS Metal-based applications.
//!
//! This class provides simple C++ wrapper around Metal API. Here's a minimal
//! example that shows how to create and launch an Metal-based app:
//!
//! \code{.cpp}
//! #include <jet.gfx/jet.gfx.h>
//!
//! int main() {
//!     MetalApp::initialize();
//!     auto window = MetalApp::createWindow("Metal Tests", 1280, 720);
//!     MetalApp::run();
//! }
//!
//! \endcode
//!
class MetalApp {
 public:
    //! Initializes the app.
    static int initialize();

    //! Starts the run-loop.
    static int run();

    //!
    //! Creates a Metal window.
    //!
    //! \param title    Title of the window.
    //! \param width    Width of the window.
    //! \param height   Height of the window.
    //!
    //! \return Metal Window object.
    //!
    static MetalWindowPtr createWindow(const std::string& title, int width,
                                       int height);
};

}  // namespace gfx
}  // namespace jet

#endif  // JET_MACOSX

#endif  // INCLUDE_JET_GFX_METAL_APP_H_
