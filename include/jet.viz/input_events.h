// Copyright (c) 2017 Doyub Kim
//
// I am making my contributions/submissions to this project solely in my
// personal capacity and am not conveying any rights to any intellectual
// property of any third parties.

#ifndef INCLUDE_JET_VIZ_INPUT_EVENTS_H_
#define INCLUDE_JET_VIZ_INPUT_EVENTS_H_

#include <jet/macros.h>
#include <cstdint>

namespace jet { namespace viz {

enum class PointerInputType : int8_t {
    Unknown = 0,
    Mouse = 1,
    Touch = 2,
    Pen = 3,
};

enum class MouseButtonType : int8_t {
    None = 0,
    Left = 1,
    Middle = 2,
    Right = 3,
};

enum class ModifierKey : int8_t {
    None = 0,
    Shift = 1 << 0,
    Ctrl = 1 << 1,
    Alt = 1 << 2,
};

inline ModifierKey operator&(ModifierKey a, ModifierKey b) {
    return static_cast<ModifierKey>(static_cast<int>(a) & static_cast<int>(b));
}

inline ModifierKey operator|(ModifierKey a, ModifierKey b) {
    return static_cast<ModifierKey>(static_cast<int>(a) | static_cast<int>(b));
}

enum class SpecialKey : int8_t {
    None = 0,
    F1,
    F2,
    F3,
    F4,
    F5,
    F6,
    F7,
    F8,
    F9,
    F10,
    F11,
    F12,
    Left,
    Up,
    Right,
    Down,
    PageUp,
    PageDown,
    Home,
    End,
    Insert,
};

struct MouseWheelData {
    double deltaX = 0.0;
    double deltaY = 0.0;
};

class PointerEvent {
 public:
    PointerEvent();

    PointerEvent(PointerInputType newInputType, ModifierKey newModifierKey,
                 double newX, double newY, double newDeltaX, double newDelyaY,
                 MouseButtonType pressedMouseButton, MouseWheelData wheelData);

    PointerInputType inputType() const;
    ModifierKey modifierKey() const;
    double x() const;
    double y() const;
    double deltaX() const;
    double deltaY() const;

    MouseButtonType pressedMouseButton() const;
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

class KeyEvent {
 public:
    KeyEvent();

    KeyEvent(int newKey, SpecialKey newSpecialKey, ModifierKey newModifierKey);

    int key() const;
    SpecialKey specialKey() const;
    ModifierKey modifierKey() const;

 private:
    int _key = 0;
    SpecialKey _specialKey = SpecialKey::None;
    ModifierKey _modifierKey = ModifierKey::None;
};

} }  // namespace jet::viz

#endif  // INCLUDE_JET_VIZ_INPUT_EVENTS_H_
