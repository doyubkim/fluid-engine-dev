// Copyright (c) 2018 Doyub Kim
//
// I am making my contributions/submissions to this project solely in my
// personal capacity and am not conveying any rights to any intellectual
// property of any third parties.

#ifndef SRC_EXAMPLES_COMMON_EXAMPLE_H_
#define SRC_EXAMPLES_COMMON_EXAMPLE_H_

#include <jet.viz/jet.viz.h>
#include <jet/animation.h>

#include <string>

class Example {
 public:
    explicit Example(const jet::Frame& frame);
    virtual ~Example() = default;

    virtual std::string name() const = 0;

#ifdef JET_USE_GL
    void setup(jet::viz::GlfwWindow* window);

    void gui(jet::viz::GlfwWindow* window);
#else
    void setup();
#endif

    //! Rewinds to frame zero, but keeping control parameters the same.
    void restartSim();

    //! Advances sim state from worker thread.
    void advanceSim();

    //! Updates renderables from main thread.
    void updateRenderables();

 protected:
    virtual void onRestartSim();

#ifdef JET_USE_GL
    virtual void onSetup(jet::viz::GlfwWindow* window);

    virtual void onGui(jet::viz::GlfwWindow* window);
#else
    virtual void onSetup();
#endif

    virtual void onAdvanceSim(const jet::Frame& frame);

    virtual void onUpdateRenderables();

 private:
    jet::Frame _frame;
};

typedef std::shared_ptr<Example> ExamplePtr;

#endif  // SRC_EXAMPLES_COMMON_EXAMPLE_H_
