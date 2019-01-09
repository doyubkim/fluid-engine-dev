// Copyright (c) 2018 Doyub Kim
//
// I am making my contributions/submissions to this project solely in my
// personal capacity and am not conveying any rights to any intellectual
// property of any third parties.

#include "gfx_example_manager.h"

#include <jet.gfx/jet.gfx.h>
#include <jet/jet.h>

using namespace jet;
using namespace gfx;

namespace {

WindowPtr sWindow;
Array1<GfxExamplePtr> sTests;
size_t sCurrentTestIdx = 0;

void toggleUpdate(Window* win, bool onoff) { win->setIsUpdateEnabled(onoff); }

void toggleUpdate(Window* win) { toggleUpdate(win, !win->isUpdateEnabled()); }

void nextTests(Window* win) {
    sWindow->renderer()->clearRenderables();

    toggleUpdate(win, false);

    ++sCurrentTestIdx;
    if (sCurrentTestIdx == sTests.length() || sTests.isEmpty()) {
        sCurrentTestIdx = 0;
    }

    if (sTests.length() > 0) {
        sTests[sCurrentTestIdx]->setup(sWindow.get());
    }
}

bool onKeyDown(Window* win, const KeyEvent& keyEvent) {
    int key = keyEvent.key();

    // "Enter" key for toggling animation.
    // "Space" key for moving to the next example.
    if (key == 13) {
        toggleUpdate(win);
        return true;
    } else if (key == ' ') {
        nextTests(win);
        JET_INFO << "Starting example ID: " << sCurrentTestIdx;
        return true;
    } else if (key == '>' || key == '.') {
        JET_INFO << "Advancing frame to "
                 << sTests[sCurrentTestIdx]->currentFrame().index + 1;
        sTests[sCurrentTestIdx]->advanceSim();
    } else if (key == 'r' || key == 'R') {
        sTests[sCurrentTestIdx]->restartSim();
    }

    return false;
}

bool onUpdate(Window* win) {
    sTests[sCurrentTestIdx]->advanceSim();
    return true;
}

}  // namespace

void GfxExampleManager::initialize(const jet::gfx::WindowPtr& window) {
    sWindow = window;

    sWindow->onKeyDownEvent() += onKeyDown;
    sWindow->onUpdateEvent() += onUpdate;
}

void GfxExampleManager::addExample(const GfxExamplePtr& example) {
    sTests.append(example);

    // Setup the first example.
    if (sTests.length() == 1) {
        example->setup(sWindow.get());
    }
}
