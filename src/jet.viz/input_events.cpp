// Copyright (c) 2017 Doyub Kim
//
// I am making my contributions/submissions to this project solely in my
// personal capacity and am not conveying any rights to any intellectual
// property of any third parties.

#include <pch.h>

#include <jet.viz/input_events.h>

using namespace jet;
using namespace viz;

// PointerEvent implementation
PointerEvent::PointerEvent()
    : _inputType(PointerInputType::Unknown),
      _modifierKey(ModifierKey::None),
      _x(0.0),
      _y(0.0),
      _deltaX(0.0),
      _deltaY(0.0),
      _pressedMouseButton(MouseButtonType::None) {}

PointerEvent::PointerEvent(PointerInputType newInputType,
                           ModifierKey newModifierKey, double newX, double newY,
                           double newDeltaX, double newDelyaY,
                           MouseButtonType pressedMouseButton,
                           MouseWheelData wheelData)
    : _inputType(newInputType),
      _modifierKey(newModifierKey),
      _x(newX),
      _y(newY),
      _deltaX(newDeltaX),
      _deltaY(newDelyaY),
      _pressedMouseButton(pressedMouseButton),
      _wheelData(wheelData) {}

PointerInputType PointerEvent::inputType() const { return _inputType; }

ModifierKey PointerEvent::modifierKey() const { return _modifierKey; }

double PointerEvent::x() const { return _x; }

double PointerEvent::y() const { return _y; }

double PointerEvent::deltaX() const { return _deltaX; }

double PointerEvent::deltaY() const { return _deltaY; }

MouseButtonType PointerEvent::pressedMouseButton() const {
    return _pressedMouseButton;
}

const MouseWheelData& PointerEvent::wheelData() const { return _wheelData; }

// KeyEvent implementation
KeyEvent::KeyEvent()
    : _key(0), _specialKey(SpecialKey::None), _modifierKey(ModifierKey::None) {}

KeyEvent::KeyEvent(int newKey, SpecialKey newSpecialKey,
                   ModifierKey newModifierKey)
    : _key(newKey), _specialKey(newSpecialKey), _modifierKey(newModifierKey) {}

int KeyEvent::key() const { return _key; }

SpecialKey KeyEvent::specialKey() const { return _specialKey; }

ModifierKey KeyEvent::modifierKey() const { return _modifierKey; }
