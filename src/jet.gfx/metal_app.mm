// Copyright (c) 2018 Doyub Kim
//
// I am making my contributions/submissions to this project solely in my
// personal capacity and am not conveying any rights to any intellectual
// property of any third parties.

#include <jet.gfx/metal_app.h>
#include <jet.gfx/metal_window.h>

#include "3rdparty/mtlpp/mtlpp.hpp"

#import <Cocoa/Cocoa.h>

namespace jet {

namespace gfx {

int MetalApp::initialize() {
    return 0;
}

int MetalApp::run() {
    NSApplication *application = [NSApplication sharedApplication];
    (void) application;
    [NSApp setActivationPolicy:NSApplicationActivationPolicyRegular];

    NSMenu *menubar = [NSMenu new];
    NSMenuItem *appMenuItem = [NSMenuItem new];
    NSMenu *appMenu = [NSMenu new];
    NSMenuItem *quitMenuItem = [[NSMenuItem alloc] initWithTitle:@"Quit" action:@selector(stop:) keyEquivalent:@"q"];
    [menubar addItem:appMenuItem];
    [appMenu addItem:quitMenuItem];
    [appMenuItem setSubmenu:appMenu];
    [NSApp setMainMenu:menubar];

    [NSApp activateIgnoringOtherApps:YES];
    [NSApp run];

    return 0;
}

MetalWindowPtr MetalApp::createWindow(const std::string &title, int width,
                                      int height) {
    return MetalWindowPtr(new MetalWindow(title, width, height));
}

}

}
