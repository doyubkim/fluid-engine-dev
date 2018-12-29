// Copyright (c) 2018 Doyub Kim
//
// I am making my contributions/submissions to this project solely in my
// personal capacity and am not conveying any rights to any intellectual
// property of any third parties.

#include <common.h>

#include <jet.gfx/texture2.h>

namespace jet {
namespace gfx {

Texture2::Texture2() {}

Texture2::~Texture2() {}

void Texture2::clear() {
    _size = Vector2UZ();

    onClear();
}

void Texture2::setTexture(const ConstArrayView2<Vector4F> &data) {
    if (data.size() == Vector2UZ()) {
        clear();
    } else if (data.size() == _size) {
        update(data);
    } else {
        clear();

        _size = data.size();

        onSetTexture(data);
    }
}

void Texture2::setTexture(const ConstArrayView2<Vector4B> &data) {
    if (data.size() == Vector2UZ()) {
        clear();
    } else if (data.size() == _size) {
        update(data);
    } else {
        clear();

        _size = data.size();

        onSetTexture(data);
    }
}

void Texture2::bind(Renderer *renderer, unsigned int slotId) {
    onBind(renderer, slotId);
}

const Vector2UZ &Texture2::size() const { return _size; }

const TextureSamplingMode &Texture2::samplingMode() const {
    return _samplingMode;
}

void Texture2::setSamplingMode(const TextureSamplingMode &mode) {
    _samplingMode = mode;
    onSamplingModeChanged(mode);
}

}  // namespace gfx
}  // namespace jet
