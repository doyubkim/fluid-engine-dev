// Copyright (c) 2018 Doyub Kim
//
// I am making my contributions/submissions to this project solely in my
// personal capacity and am not conveying any rights to any intellectual
// property of any third parties.

#undef JET_USE_GL
#import <common.h>

#import <jet.gfx/metal_window.h>
#import <jet.gfx/metal_renderer.h>

#import "metal_view.h"
#import "mtlpp_wrappers.h"

#import <Cocoa/Cocoa.h>
#import <MetalKit/MetalKit.h>

// MARK: WindowViewController

@interface WindowViewController : NSViewController<MTKViewDelegate> {
 @public
    void (*render)(const jet::gfx::MetalWindow&);
 @public
    const jet::gfx::MetalWindow* window;
}

@end

@implementation WindowViewController
- (void)mtkView:(nonnull MTKView*)view drawableSizeWillChange:(CGSize)size {
}

- (void)drawInMTKView:(nonnull MTKView*)view {
    (*render)(*window);
}
@end

// MARK: MetalWindow

namespace jet {

namespace gfx {

MetalWindow::MetalWindow(const std::string &title, int width, int height) {
    auto renderer = std::make_shared<MetalRenderer>(this);

    NSRect frame = NSMakeRect(0, 0, width, height);
    NSWindow *window =
            [[NSWindow alloc] initWithContentRect:frame
#if MTLPP_IS_AVAILABLE_MAC(10_12)
                                        styleMask:(NSWindowStyleMaskTitled |
                                                   NSWindowStyleMaskClosable |
                                                   NSWindowStyleMaskResizable)
#else
            styleMask:(NSTitledWindowMask |
                       NSClosableWindowMask |
                       NSResizableWindowMask)
#endif
                                          backing:NSBackingStoreBuffered
                                            defer:NO];
    window.title = [[NSProcessInfo processInfo] processName];
    WindowViewController *viewController = [WindowViewController new];
    viewController->render = MetalWindow::render;
    viewController->window = this;

    MTKView *view = [[MTKView alloc] initWithFrame:frame];
    view.device = (__bridge id <MTLDevice>) renderer->device()->value.GetPtr();
    view.delegate = viewController;
    view.autoresizingMask = NSViewWidthSizable | NSViewHeightSizable;

    [window.contentView addSubview:view];
    [window center];
    [window orderFrontRegardless];

    _view = new MetalView(ns::Handle{(__bridge void *) view});

    setRenderer(renderer);
}

MetalWindow::~MetalWindow() { delete _view; }

void MetalWindow::setSwapInterval(int interval) {
    // TODO: Implement
    UNUSED_VARIABLE(interval);
}

Vector2UZ MetalWindow::framebufferSize() const {
    // TODO: Implement
    return Vector2UZ();
}

Vector2UZ MetalWindow::windowSize() const {
    // TODO: Implement
    return Vector2UZ();
}

void MetalWindow::requestRender(unsigned int numFrames) {
    // TODO: Implement
    UNUSED_VARIABLE(numFrames);
}

MetalView *MetalWindow::view() const { return _view; }

/* static */ void MetalWindow::render(const MetalWindow &window) {
    std::static_pointer_cast<MetalRenderer>(window.renderer())->render();
}

}

}