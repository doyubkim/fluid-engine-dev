// Copyright (c) 2017 Doyub Kim
//
// I am making my contributions/submissions to this project solely in my
// personal capacity and am not conveying any rights to any intellectual
// property of any third parties.

#include <pch.h>

#include <jet.viz/image.h>

#define STB_IMAGE_IMPLEMENTATION
#include <stb/stb_image.h>

using namespace jet;
using namespace viz;

ByteImage::ByteImage() {}

ByteImage::ByteImage(size_t width, size_t height,
                     const ByteColor& initialValue) {
    _data.resize(width, height, initialValue);
}

ByteImage::ByteImage(const std::string& filename) {
    int width, height, bpp;
    stbi_set_flip_vertically_on_load(true);
    unsigned char* rawData =
        stbi_load(filename.c_str(), &width, &height, &bpp, 4);
    const auto swidth = static_cast<size_t>(width);
    const auto sheight = static_cast<size_t>(height);

    _data.resize(swidth, sheight);

    size_t c = 0;
    ByteColor color;
    for (size_t j = 0; j < sheight; ++j) {
        for (size_t i = 0; i < swidth; ++i) {
            color.r = rawData[c++];
            color.g = rawData[c++];
            color.b = rawData[c++];
            color.a = rawData[c++];
            _data(i, j) = color;
        }
    }

    stbi_image_free(rawData);
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
