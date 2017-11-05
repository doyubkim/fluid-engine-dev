// Copyright (c) 2017 Doyub Kim
//
// I am making my contributions/submissions to this project solely in my
// personal capacity and am not conveying any rights to any intellectual
// property of any third parties.

#include <pch.h>

#include <jet.viz/texture2.h>

using namespace jet;
using namespace viz;

Texture2::Texture2() {}

Texture2::~Texture2() {}

void Texture2::clear() {
    _size = Size2();

    onClear();
}

void Texture2::resize(const float* const data, const Size2& size) {
    if (size == Size2()) {
        clear();
    } else if (size == _size) {
        update(data);
    } else {
        clear();

        _size = size;

        onResize(data, size);
    }
}

void Texture2::resize(const uint8_t* const data, const Size2& size) {
    if (size == Size2()) {
        clear();
    } else if (size == _size) {
        update(data);
    } else {
        clear();

        _size = size;

        onResize(data, size);
    }
}

void Texture2::bind(Renderer* renderer, unsigned int slotId) {
    onBind(renderer, slotId);
}

const Size2& Texture2::size() const { return _size; }

const TextureSamplingMode& Texture2::samplingMode() const {
    return _samplingMode;
}

void Texture2::setSamplingMode(const TextureSamplingMode& mode) {
    _samplingMode = mode;
    onSamplingModeChanged(mode);
}
