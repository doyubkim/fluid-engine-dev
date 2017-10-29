// Copyright (c) 2017 Doyub Kim
//
// I am making my contributions/submissions to this project solely in my
// personal capacity and am not conveying any rights to any intellectual
// property of any third parties.

#include <pch.h>

#include <jet.viz/image.h>

using namespace jet;
using namespace viz;

ByteImage::ByteImage() {}

ByteImage::ByteImage(size_t width, size_t height,
                     const ByteColor& initialValue) {
    _data.resize(width, height, initialValue);
}

void ByteImage::clear() {}

void ByteImage::resize(size_t width, size_t height,
                       const ByteColor& initialValue) {
    _data.resize(width, height, initialValue);
}

Size2 ByteImage::size() const { return _data.size(); }

ByteColor* ByteImage::data() { return _data.data(); }

const ByteColor* const ByteImage::data() const { return _data.data(); }

ByteColor& ByteImage::operator()(size_t i, size_t j) { return _data(i, j); }

const ByteColor& ByteImage::operator()(size_t i, size_t j) const {
    return _data(i, j);
}
