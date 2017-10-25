// Copyright (c) 2017 Doyub Kim
//
// I am making my contributions/submissions to this project solely in my
// personal capacity and am not conveying any rights to any intellectual
// property of any third parties.

#ifndef INCLUDE_JET_VIZ_IMAGE_H_
#define INCLUDE_JET_VIZ_IMAGE_H_

#include <jet/array2.h>
#include <jet.viz/color.h>

#include <memory>

namespace jet { namespace viz {

class ByteImage final {
 public:
    ByteImage();

    explicit ByteImage(std::size_t width, std::size_t height,
                       const ByteColor& initialValue = ByteColor());

    void clear();

    void resize(std::size_t width, std::size_t height,
                const ByteColor& initialValue = ByteColor());

    Size2 size() const;

    ByteColor* data();

    const ByteColor* const data() const;

    ByteColor& operator()(std::size_t i, std::size_t j);

    const ByteColor& operator()(std::size_t i, std::size_t j) const;

 private:
    Array2<ByteColor> _data;
};

typedef std::shared_ptr<ByteImage> ByteImagePtr;

} }  // namespace jet::viz

#endif  // INCLUDE_JET_VIZ_IMAGE_H_
