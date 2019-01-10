// Copyright (c) 2018 Doyub Kim
//
// I am making my contributions/submissions to this project solely in my
// personal capacity and am not conveying any rights to any intellectual
// property of any third parties.
//
// This code uses example code from mtlpp
// (https://github.com/naleksiev/mtlpp)
// This code also uses key handling code from ImGui
// (https://github.com/ocornut/imgui)
//

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

class MetalCustomViewEventHandler {
 public:
    static void onRender(MetalWindow *window) { window->onRender(); }

    static bool onEvent(MetalWindow *window, NSEvent *event, NSView *view) {
        ModifierKey mods = getModifier(event.modifierFlags);

        bool handled = false;

        NSPoint posInWin = event.locationInWindow;
        NSPoint pos = [view convertPoint:posInWin fromView:nil];

        if (event.type == NSEventTypeKeyDown) {
            NSString *str = event.characters;
            for (int i = 0; i < str.length; i++) {
                int c = [str characterAtIndex:i];
                int key = mapCharacterToKey(c);
                handled |= window->onKeyDown(KeyEvent(key, mods));
            }
        } else if (event.type == NSEventTypeKeyUp) {
            NSString *str = event.characters;
            for (int i = 0; i < str.length; i++) {
                int c = [str characterAtIndex:i];
                int key = mapCharacterToKey(c);
                handled |= window->onKeyUp(KeyEvent(key, mods));
            }
        } else if (event.type == NSEventTypeLeftMouseDown ||
                   event.type == NSEventTypeRightMouseDown ||
                   event.type == NSEventTypeOtherMouseDown) {
            MouseButtonType newButtonType = getMouseButton(event.type);
            window->onMouseDown(newButtonType, mods, (float)pos.x,
                                (float)pos.y);

        } else if (event.type == NSEventTypeLeftMouseUp ||
                   event.type == NSEventTypeRightMouseUp ||
                   event.type == NSEventTypeOtherMouseUp) {
            MouseButtonType newButtonType = getMouseButton(event.type);
            window->onMouseUp(newButtonType, mods, (float)pos.x, (float)pos.y);
        } else if (event.type == NSEventTypeLeftMouseDragged ||
                   event.type == NSEventTypeRightMouseDragged ||
                   event.type == NSEventTypeOtherMouseDragged) {
            MouseButtonType newButtonType = getMouseButton(event.type);
            window->onMouseDragged(newButtonType, mods, (float)pos.x,
                                   (float)pos.y, (float)event.deltaX,
                                   (float)event.deltaY);
        } else if (event.type == NSEventTypeMouseMoved) {
            window->onMouseHover(mods, (float)pos.x, (float)pos.y,
                                 (float)event.deltaX, (float)event.deltaY);
        } else if (event.type == NSEventTypeMouseEntered) {
            window->onMouseEntered(true);
        } else if (event.type == NSEventTypeMouseExited) {
            window->onMouseEntered(false);
        } else if (event.type == NSEventTypeScrollWheel) {
            window->onMouseScrollWheel(mods, (float)pos.x, (float)pos.y,
                                       (float)event.deltaX,
                                       (float)event.deltaY);
        }

        if (event.type != NSEventTypeMouseMoved || handled) {
            window->requestRender(1);
        }

        return handled;
    }

    static void onWindowResized(MetalWindow *window, CGFloat w, CGFloat h) {
        window->onWindowResized((int)w, (int)h);
    }

    static void onWindowMoved(MetalWindow *window, CGFloat x, CGFloat y) {
        window->onWindowMoved(x, y);
    }

    static int mapCharacterToKey(int c) {
        // Translate the key to Straight ASCII
        if (c >= 'a' && c <= 'z') {
            return c - 'a' + 'A';
        }
        if (c == 25) {  // SHIFT+TAB -> TAB
            return 9;
        }
        if (c >= 0 && c < 256) {
            return c;
        }
        if (c >= 0xF700 && c < 0xF700 + 256) {
            return c - 0xF700 + 256;
        }
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

    static MouseButtonType getMouseButton(NSEventType eventType) {
        switch (eventType) {
            case NSEventTypeLeftMouseDown:
            case NSEventTypeLeftMouseUp:
            case NSEventTypeLeftMouseDragged:
                return MouseButtonType::Left;
            case NSEventTypeRightMouseDown:
            case NSEventTypeRightMouseUp:
            case NSEventTypeRightMouseDragged:
                return MouseButtonType::Right;
            case NSEventTypeOtherMouseDown:
            case NSEventTypeOtherMouseUp:
            case NSEventTypeOtherMouseDragged:
                // TODO: Translating to middle button which is not ideal.
                return MouseButtonType::Middle;
            default:
                return MouseButtonType::None;
        }
    }
};
}
}

// MARK: MetalCustomWindowDelegate

@interface MetalCustomWindowDelegate : NSObject<NSWindowDelegate> {
 @public
    jet::gfx::MetalWindow *window;
}
@end

@implementation MetalCustomWindowDelegate
- (void)windowDidMove:(NSNotification *)notification {
    NSWindow *nsWindow = (__bridge NSWindow *)window->window()->GetPtr();
    jet::gfx::MetalCustomViewEventHandler::onWindowMoved(
        window, nsWindow.frame.origin.x, nsWindow.frame.origin.y);
}
@end

// MARK: MetalCustomView

@interface MetalCustomView : MTKView {
 @public
    jet::gfx::MetalWindow *window;
}

@end

@implementation MetalCustomView

- (id)initWithFrame:(NSRect)frame {
    self = [super initWithFrame:frame];
    if (self) {
        //
    }
    return self;
}

- (BOOL)acceptsFirstResponder {
    return YES;
}

- (void)keyDown:(NSEvent *)event {
    jet::gfx::MetalCustomViewEventHandler::onEvent(window, event, self);
}

- (void)keyUp:(NSEvent *)event {
    jet::gfx::MetalCustomViewEventHandler::onEvent(window, event, self);
}

- (void)mouseDown:(NSEvent *)event {
    jet::gfx::MetalCustomViewEventHandler::onEvent(window, event, self);
}

- (void)mouseUp:(NSEvent *)event {
    jet::gfx::MetalCustomViewEventHandler::onEvent(window, event, self);
}

- (void)mouseMoved:(NSEvent *)event {
    jet::gfx::MetalCustomViewEventHandler::onEvent(window, event, self);
}

- (void)mouseDragged:(NSEvent *)event {
    jet::gfx::MetalCustomViewEventHandler::onEvent(window, event, self);
}

- (void)mouseEntered:(NSEvent *)event {
    jet::gfx::MetalCustomViewEventHandler::onEvent(window, event, self);
}

- (void)mouseExited:(NSEvent *)event {
    jet::gfx::MetalCustomViewEventHandler::onEvent(window, event, self);
}

- (void)scrollWheel:(NSEvent *)event {
    jet::gfx::MetalCustomViewEventHandler::onEvent(window, event, self);
}

@end

// MARK: MetalCustomViewController

@interface MetalCustomViewController : NSViewController<MTKViewDelegate> {
 @public
    jet::gfx::MetalWindow *window;
}

@end

@implementation MetalCustomViewController

- (void)mtkView:(nonnull MTKView *)view drawableSizeWillChange:(CGSize)size {
    // `size` is scaled size
    // So scale it down to "window" size
    jet::Vector2F scale = window->displayScalingFactor();
    jet::gfx::MetalCustomViewEventHandler::onWindowResized(
        window, size.width / scale.x, size.height / scale.y);
}

- (void)drawInMTKView:(nonnull MTKView *)view {
    jet::gfx::MetalCustomViewEventHandler::onRender(window);
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
    MetalCustomViewController *viewController = [MetalCustomViewController new];
    viewController->window = this;
    MetalCustomView *view = [[MetalCustomView alloc] initWithFrame:frame];
    view->window = this;
    view.device = (__bridge id<MTLDevice>)renderer->device()->value.GetPtr();
    view.delegate = viewController;
    view.autoresizingMask = NSViewWidthSizable | NSViewHeightSizable;
    view.colorPixelFormat = MTLPixelFormatBGRA8Unorm;
    view.depthStencilPixelFormat = MTLPixelFormatDepth32Float;

    // By default, render only when needed
    view.paused = YES;
    view.enableSetNeedsDisplay = YES;

    [window.contentView addSubview:view];
    [window center];
    [window orderFrontRegardless];

    _view = new MetalPrivateView(ns::Handle{(__bridge void *)view});
    _window = new MetalPrivateWindow(ns::Handle{(__bridge void *)window});

    MetalCustomWindowDelegate *windowDelegate = [MetalCustomWindowDelegate new];
    windowDelegate->window = this;
    window.delegate = windowDelegate;

    JET_INFO << "Metal window created with " << view.device.name.UTF8String;

    setRenderer(renderer);

    setViewController(std::make_shared<PitchYawViewController>(
        std::make_shared<PerspCamera>(), Vector3F()));

    onWindowResized(width, height);

    requestRender(1);
}

MetalWindow::~MetalWindow() { delete _view; }

void MetalWindow::onUpdateEnabled(bool enabled) {
    MTKView *mtkView = (__bridge MTKView *)_view->GetPtr();
    mtkView.paused = !enabled;
}

void MetalWindow::setSwapInterval(int interval) {
    MTKView *mtkView = (__bridge MTKView *)_view->GetPtr();
    // TODO: This is quite a hacky solution..
    int fps = 120 >> interval;
    mtkView.preferredFramesPerSecond = fps;
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
    MTKView *mtkView = (__bridge MTKView *)_view->GetPtr();
    NSSize size = NSSizeFromCGSize(CGSizeMake(_width, _height));
    NSSize backingSize = [mtkView convertSizeToBacking:size];

    return Vector2F(static_cast<float>(backingSize.width / size.width),
                    static_cast<float>(backingSize.height / size.height));
}

void MetalWindow::requestRender(unsigned int numFrames) {
    _numRequestedRenderFrames = numFrames;

    MTKView *mtkView = (MTKView *)_view->GetPtr();
    mtkView.needsDisplay = YES;
}

MetalPrivateWindow *MetalWindow::window() const { return _window; }

MetalPrivateView *MetalWindow::view() const { return _view; }

void MetalWindow::onRender() {
    if (isUpdateEnabled() || _numRequestedRenderFrames > 0) {
        if (isUpdateEnabled()) {
            onUpdate();
        }

        JET_ASSERT(renderer());

        renderer()->render();

        onGuiEvent()(this);

        // Decrease render request count
        --_numRequestedRenderFrames;
    }
}

bool MetalWindow::onWindowResized(int width, int height) {
    JET_ASSERT(renderer());

    Vector2F scaleFactor = displayScalingFactor();

    Viewport viewport;
    viewport.x = 0.0;
    viewport.y = 0.0;
    viewport.width = scaleFactor.x * width;
    viewport.height = scaleFactor.y * height;

    _width = width;
    _height = height;

    viewController()->setViewport(viewport);

    return onWindowResizedEvent()(this, {width, height});
}

bool MetalWindow::onWindowMoved(int x, int y) {
    return onWindowMovedEvent()(this, {x, y});
}

bool MetalWindow::onUpdate() {
    // Update
    return onUpdateEvent()(this);
}

bool MetalWindow::onKeyDown(const KeyEvent &keyEvent) {
    viewController()->keyDown(keyEvent);
    return onKeyDownEvent()(this, keyEvent);
}

bool MetalWindow::onKeyUp(const KeyEvent &keyEvent) {
    viewController()->keyUp(keyEvent);
    return onKeyUpEvent()(this, keyEvent);
}

bool MetalWindow::onMouseDown(MouseButtonType button, ModifierKey mods, float x,
                              float y) {
    Vector2F scale = displayScalingFactor();
    x *= scale.x;
    y *= scale.y;
    PointerEvent pointerEvent(PointerInputType::Mouse, mods, x, y, 0, 0, button,
                              MouseWheelData());
    viewController()->pointerPressed(pointerEvent);
    return onPointerPressedEvent()(this, pointerEvent);
}

bool MetalWindow::onMouseUp(MouseButtonType button, ModifierKey mods, float x,
                            float y) {
    Vector2F scale = displayScalingFactor();
    x *= scale.x;
    y *= scale.y;
    PointerEvent pointerEvent(PointerInputType::Mouse, mods, x, y, 0, 0, button,
                              MouseWheelData());
    viewController()->pointerReleased(pointerEvent);
    return onPointerReleasedEvent()(this, pointerEvent);
}

bool MetalWindow::onMouseDragged(MouseButtonType button, ModifierKey mods,
                                 float x, float y, float dx, float dy) {
    Vector2F scale = displayScalingFactor();
    x *= scale.x;
    y *= scale.y;
    dx *= scale.x;
    dy *= scale.y;
    PointerEvent pointerEvent(PointerInputType::Mouse, mods, x, y, dx, dy,
                              button, MouseWheelData());
    viewController()->pointerDragged(pointerEvent);
    return onPointerDraggedEvent()(this, pointerEvent);
}

bool MetalWindow::onMouseHover(ModifierKey mods, float x, float y, float dx,
                               float dy) {
    Vector2F scale = displayScalingFactor();
    x *= scale.x;
    y *= scale.y;
    dx *= scale.x;
    dy *= scale.y;
    PointerEvent pointerEvent(PointerInputType::Mouse, mods, x, y, dx, dy,
                              MouseButtonType::None, MouseWheelData());
    viewController()->pointerHover(pointerEvent);
    return onPointerHoverEvent()(this, pointerEvent);
}

bool MetalWindow::onMouseScrollWheel(ModifierKey mods, float x, float y,
                                     float dx, float dy) {
    Vector2F scale = displayScalingFactor();
    x *= scale.x;
    y *= scale.y;
    dx *= scale.x;
    dy *= scale.y;

    MouseWheelData wheelData;
    wheelData.deltaX = dx;
    wheelData.deltaY = dy;

    PointerEvent pointerEvent(PointerInputType::Mouse, mods, x, y, dx, dy,
                              MouseButtonType::None, wheelData);
    viewController()->mouseWheel(pointerEvent);
    return onMouseWheelEvent()(this, pointerEvent);
}

bool MetalWindow::onMouseEntered(bool entered) {
    return onPointerEnterEvent()(this, entered);
}
}
}