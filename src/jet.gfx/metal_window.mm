// Copyright (c) 2018 Doyub Kim
//
// I am making my contributions/submissions to this project solely in my
// personal capacity and am not conveying any rights to any intellectual
// property of any third parties.
//
// This code uses example code from mtlpp
// (https://github.com/naleksiev/mtlpp)
// and imgui
// (https://github.com/ocornut/imgui)

#undef JET_USE_GL
#import <common.h>

#import <jet.gfx/metal_renderer.h>
#import <jet.gfx/metal_window.h>
#import <jet.gfx/persp_camera.h>
#import <jet.gfx/pitch_yaw_view_controller.h>

#import "metal_view.h"
#import "mtlpp_wrappers.h"

#import <Cocoa/Cocoa.h>
#import <MetalKit/MetalKit.h>

namespace jet {
namespace gfx {

class MetalWindowEventHandler {
 public:
    static void onRender(MetalWindow *window) { window->onRender(); }

    static bool onEvent(MetalWindow *window, NSEvent *event, NSView *view) {
        ModifierKey mods = getModifier(event.modifierFlags);

        if (event.type == NSEventTypeKeyDown) {
            // TODO: Move under MetalWindow
            bool handled = false;
            NSString *str = event.characters;
            for (int i = 0; i < str.length; i++) {
                int c = [str characterAtIndex:i];
                int key = mapCharacterToKey(c);
                KeyEvent keyEvent(key, mods);
                window->viewController()->keyDown(keyEvent);
                handled |= window->onKeyDownEvent()(window, keyEvent);
            }
            if (handled) {
                window->requestRender(1);
            }
            return handled;
        } else if (event.type == NSEventTypeKeyUp) {
            // TODO: Move under MetalWindow
            bool handled = false;
            NSString *str = event.characters;
            for (int i = 0; i < str.length; i++) {
                int c = [str characterAtIndex:i];
                int key = mapCharacterToKey(c);
                KeyEvent keyEvent(key, mods);
                window->viewController()->keyUp(keyEvent);
                handled |= window->onKeyUpEvent()(window, keyEvent);
            }
            if (handled) {
                window->requestRender(1);
            }
            return handled;
        }

        return false;
    }

    static void onWindowResized(MetalWindow *window, CGFloat w, CGFloat h) {
        window->onWindowResized((int)w, (int)h);
    }

    static int mapCharacterToKey(int c) {
        if (c >= 'a' && c <= 'z') return c - 'a' + 'A';
        if (c == 25)  // SHIFT+TAB -> TAB
            return 9;
        if (c >= 0 && c < 256) return c;
        if (c >= 0xF700 && c < 0xF700 + 256) return c - 0xF700 + 256;
        return -1;
    }

    static ModifierKey getModifier(NSEventModifierFlags mods) {
        ModifierKey modifier = ModifierKey::None;

        if (mods & NSEventModifierFlagOption) {
            modifier = modifier | ModifierKey::Alt;
        }
        if (mods & NSEventModifierFlagControl) {
            modifier = modifier | ModifierKey::Ctrl;
        }
        if (mods & NSEventModifierFlagShift) {
            modifier = modifier | ModifierKey::Shift;
        }

        return modifier;
    }
};
}
}

// MARK: WindowViewController

@interface WindowViewController : NSViewController<MTKViewDelegate> {
 @public
    jet::gfx::MetalWindow *window;
}

@end

@implementation WindowViewController
- (void)setupWithMTKView:(nonnull MTKView *)view {
    // From ImGui macOS Metal example...

    // Add a tracking area in order to receive mouse events whenever the mouse
    // is within the bounds of our view
    NSTrackingArea *trackingArea = [[NSTrackingArea alloc]
        initWithRect:NSZeroRect
             options:NSTrackingMouseMoved | NSTrackingInVisibleRect |
                     NSTrackingActiveAlways
               owner:self
            userInfo:nil];
    [view addTrackingArea:trackingArea];

    // If we want to receive key events, we either need to be in the responder
    // chain of the key view, or else we can install a local monitor. The
    // consequence of this heavy-handed approach is that we receive events for
    // all controls, not just Dear ImGui widgets. If we had native controls in
    // our window, we'd want to be much more careful than just ingesting the
    // complete event stream, though we do make an effort to be good citizens by
    // passing along events when Dear ImGui doesn't want to capture.
    NSEventMask eventMask = NSEventMaskKeyDown | NSEventMaskKeyUp |
                            NSEventMaskFlagsChanged | NSEventTypeScrollWheel;
    [NSEvent
        addLocalMonitorForEventsMatchingMask:eventMask
                                     handler:^NSEvent *_Nullable(
                                         NSEvent *event) {
                                       BOOL handled =
                                           jet::gfx::MetalWindowEventHandler::
                                               onEvent(window, event, view);
                                       // TODO: ImGui_ImplOSX_HandleEvent goes
                                       // here
                                       if (event.type == NSEventTypeKeyDown &&
                                           handled) {
                                           return nil;
                                       } else {
                                           return event;
                                       }

                                     }];
}

- (void)mtkView:(nonnull MTKView *)view drawableSizeWillChange:(CGSize)size {
    std::cout << size.width << " x " << size.height << std::endl;
    jet::gfx::MetalWindowEventHandler::onWindowResized(window, size.width,
                                                       size.height);
}

- (void)drawInMTKView:(nonnull MTKView *)view {
    jet::gfx::MetalWindowEventHandler::onRender(window);
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
    window.title = [NSString stringWithUTF8String:title.c_str()];
    WindowViewController *viewController = [WindowViewController new];
    viewController->window = this;

    MTKView *view = [[MTKView alloc] initWithFrame:frame];
    view.device = (__bridge id<MTLDevice>)renderer->device()->value.GetPtr();
    view.delegate = viewController;
    view.autoresizingMask = NSViewWidthSizable | NSViewHeightSizable;

    // By default, render only when needed
    view.paused = YES;
    view.enableSetNeedsDisplay = YES;

    [viewController setupWithMTKView:view];

    [window.contentView addSubview:view];
    [window center];
    [window orderFrontRegardless];

    _view = new MetalView(ns::Handle{(__bridge void *)view});

    JET_INFO << "Metal window created with " << view.device.name.UTF8String;

    setRenderer(renderer);

    setViewController(std::make_shared<PitchYawViewController>(
        std::make_shared<PerspCamera>(), Vector3F()));

    onWindowResized(width, height);

    requestRender(1);
}

MetalWindow::~MetalWindow() { delete _view; }

void MetalWindow::setSwapInterval(int interval) {
    // TODO: Implement
    UNUSED_VARIABLE(interval);
}

Vector2UZ MetalWindow::framebufferSize() const {
    MTKView *mtkView = (__bridge MTKView *)_view->GetPtr();
    NSSize size = NSSizeFromCGSize(CGSizeMake(_width, _height));
    NSSize backingSize = [mtkView convertSizeToBacking:size];

    return Vector2UZ((size_t)backingSize.width, (size_t)backingSize.height);
}

Vector2UZ MetalWindow::windowSize() const {
    return Vector2UZ((size_t)_width, (size_t)_height);
}

Vector2F MetalWindow::displayScalingFactor() const {
    MTKView *mtkView = (__bridge MTKView*)_view->GetPtr();
    NSSize size = NSSizeFromCGSize(CGSizeMake(_width, _height));
    NSSize backingSize = [mtkView convertSizeToBacking:size];

    return Vector2F(static_cast<float>(backingSize.width / size.width),
                    static_cast<float>(backingSize.height / size.height));
}

void MetalWindow::requestRender(unsigned int numFrames) {
    MTKView *mtkView = (MTKView *)_view->GetPtr();
    mtkView.needsDisplay = YES;
}

MetalView *MetalWindow::view() const { return _view; }

void MetalWindow::onRender() {
    JET_ASSERT(renderer());

    renderer()->render();

    onGuiEvent()(this);
}

bool MetalWindow::onWindowResized(int width, int height) {
    JET_ASSERT(renderer());

    // Scaling factor already applied
    Viewport viewport;
    viewport.x = 0.0;
    viewport.y = 0.0;
    viewport.width = width;
    viewport.height = height;

    _width = width;
    _height = height;

    viewController()->setViewport(viewport);

    return onWindowResizedEvent()(this, {width, height});
}
}
}