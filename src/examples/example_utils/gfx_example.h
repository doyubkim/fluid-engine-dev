// Copyright (c) 2018 Doyub Kim
//
// I am making my contributions/submissions to this project solely in my
// personal capacity and am not conveying any rights to any intellectual
// property of any third parties.

#ifndef SRC_EXAMPLES_EXAMPLE_UTILS_GFX_EXAMPLE_H_
#define SRC_EXAMPLES_EXAMPLE_UTILS_GFX_EXAMPLE_H_

#include <jet.gfx/jet.gfx.h>
#include <jet/jet.h>

#include <string>

class GfxExample {
 public:
    explicit GfxExample(const jet::Frame& frame);

    virtual ~GfxExample() = default;

    virtual std::string name() const = 0;

    void setup(jet::gfx::Window* window);

    void gui(jet::gfx::Window* window);

    //! Rewinds to frame zero, but keeping control parameters the same.
    void restartSim();

    //! Advances sim state from worker thread.
    void advanceSim();

 protected:
    virtual void onRestartSim();

    virtual void onSetup(jet::gfx::Window* window);

    virtual void onGui(jet::gfx::Window* window);

    virtual void onAdvanceSim(const jet::Frame& frame);

 private:
    jet::Frame _frame;
};

using GfxExamplePtr = std::shared_ptr<GfxExample>;

#endif  // SRC_EXAMPLES_EXAMPLE_UTILS_GFX_EXAMPLE_H_
