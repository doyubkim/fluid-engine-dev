// Copyright (c) 2018 Doyub Kim
//
// I am making my contributions/submissions to this project solely in my
// personal capacity and am not conveying any rights to any intellectual
// property of any third parties.

#ifndef INCLUDE_JET_VIZ_INPUT_EVENTS_H_
#define INCLUDE_JET_VIZ_INPUT_EVENTS_H_

#include <jet/macros.h>

#include <cstdint>

namespace jet {

namespace viz {

//! Pointer input types.
enum class PointerInputType : int8_t {
    //! Unknown pointer input type.
    kUnknown = 0,

    //! Mouse pointer input type.
    kMouse = 1,

    //! Touch pointer input type.
    kTouch = 2,

    //! Pen pointer input type.
    kPen = 3,
};

//! Mouse button types.
enum class MouseButtonType : int8_t {
    //! No mouse button.
    kNone = 0,

    //! Left mouse button.
    kLeft = 1,

    //! Middle mouse button.
    kMiddle = 2,

    //! Right mouse button.
    kRight = 3,
};

//! Modifier key types.
enum class ModifierKey : int8_t {
    //! No modifier.
    kNone = 0,

    //! Shift modifier key.
    kShift = 1 << 0,

    //! Ctrl modifier key.
    kCtrl = 1 << 1,

    //! Alt modifier key.
    kAlt = 1 << 2,
};

//! And operator for two modifier keys.
inline ModifierKey operator&(ModifierKey a, ModifierKey b) {
    return static_cast<ModifierKey>(static_cast<int>(a) & static_cast<int>(b));
}

//! Or operator for two modifier keys.
inline ModifierKey operator|(ModifierKey a, ModifierKey b) {
    return static_cast<ModifierKey>(static_cast<int>(a) | static_cast<int>(b));
}

//! Mouse wheel event data.
struct MouseWheelData {
    //! Horizontal scroll amount.
    double deltaX = 0.0;

    //! Vertical scroll amount.
    double deltaY = 0.0;
};

//! Pointer event representation.
class PointerEvent {
 public:
    //! Constructs an empty event.
    PointerEvent();

    //!
    //! \brief Constructs an event with parameters.
    //!
    //! \param newInputType Pointer input type.
    //! \param newModifierKey Currently pressed modifier key.
    //! \param newX X position.
    //! \param newY Y position.
    //! \param newDeltaX Delta of X from previous event.
    //! \param newDelyaY Delta of Y from previous event.
    //! \param pressedMouseButton Currently pressed mouse button.
    //! \param wheelData Mouse scroll wheel event data.
    //!
    PointerEvent(PointerInputType newInputType, ModifierKey newModifierKey,
                 double newX, double newY, double newDeltaX, double newDelyaY,
                 MouseButtonType pressedMouseButton, MouseWheelData wheelData);

    //! Returns pointer input type.
    PointerInputType inputType() const;

    //! Returns modifier key.
    ModifierKey modifierKey() const;

    //! Returns current x position.
    double x() const;

    //! Returns current y position.
    double y() const;

    //! Returns delta of x position.
    double deltaX() const;

    //! Returns delta of y position.
    double deltaY() const;

    //! Returns currently pressed mouse button.
    MouseButtonType pressedMouseButton() const;

    //! Returns mouse scroll wheel data.
    const MouseWheelData& wheelData() const;

 private:
    PointerInputType _inputType;
    ModifierKey _modifierKey;
    double _x;
    double _y;
    double _deltaX;
    double _deltaY;

    MouseButtonType _pressedMouseButton;
    MouseWheelData _wheelData;
};

//! Key input event representation.
class KeyEvent {
 public:
    //! Constructs an empty event.
    KeyEvent();

    //!
    //! \brief Constructs an event with parameters.
    //!
    //! \param newKey Key code.
    //! \param newModifierKey Modifier key type.
    //!
    KeyEvent(int newKey, ModifierKey newModifierKey);

    //! Returns key code.
    int key() const;

    //! Returns modifier key type.
    ModifierKey modifierKey() const;

 private:
    int _key = 0;
    ModifierKey _modifierKey = ModifierKey::kNone;
};

}  // namespace viz

}  // namespace jet

#endif  // INCLUDE_JET_VIZ_INPUT_EVENTS_H_
