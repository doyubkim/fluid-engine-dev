// Copyright (c) 2017 Doyub Kim
//
// I am making my contributions/submissions to this project solely in my
// personal capacity and am not conveying any rights to any intellectual
// property of any third parties.

#include <pch.h>

#include <jet.viz/texture3.h>

using namespace jet;
using namespace viz;

Texture3::Texture3() {}

Texture3::~Texture3() {}

void Texture3::clear() {
    _size = Size3();

    onClear();
}

void Texture3::resize(const float* data, const Size3& size) {
    if (size == Size3()) {
        clear();
    } else if (size == _size) {
        update(data);
    } else {
        clear();

        _size = size;

        onResize(data, size);
    }
}

void Texture3::bind(Renderer* renderer, unsigned int slotId) {
    onBind(renderer, slotId);
}

const Size3& Texture3::size() const { return _size; }

const TextureSamplingMode& Texture3::samplingMode() const {
    return _samplingMode;
}

void Texture3::setSamplingMode(const TextureSamplingMode& mode) {
    _samplingMode = mode;
    onSamplingModeChanged(mode);
}
