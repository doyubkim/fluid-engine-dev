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

void Texture3::setTexture(const ConstArrayAccessor3<ByteColor> &data) {
    if (data.size() == Size3()) {
        clear();
    } else if (data.size() == _size) {
        update(data.data());
    } else {
        clear();

        _size = data.size();

        onSetTexture(data);
    }
}

void Texture3::setTexture(const ConstArrayAccessor3<Color>& data) {
    if (data.size() == Size3()) {
        clear();
    } else if (data.size() == _size) {
        update(data.data());
    } else {
        clear();

        _size = data.size();

        onSetTexture(data);
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
