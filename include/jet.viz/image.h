// Copyright (c) 2017 Doyub Kim
//
// I am making my contributions/submissions to this project solely in my
// personal capacity and am not conveying any rights to any intellectual
// property of any third parties.

#ifndef INCLUDE_JET_VIZ_IMAGE_H_
#define INCLUDE_JET_VIZ_IMAGE_H_

#include <jet.viz/color.h>
#include <jet/array2.h>

#include <memory>

namespace jet {
namespace viz {

class ByteImage final {
 public:
    ByteImage();

    explicit ByteImage(size_t width, size_t height,
                       const ByteColor& initialValue = ByteColor());

    explicit ByteImage(const std::string& filename);

    void clear();

    void resize(size_t width, size_t height,
                const ByteColor& initialValue = ByteColor());

    Size2 size() const;

    ByteColor* data();

    const ByteColor* const data() const;

    ByteColor& operator()(size_t i, size_t j);

    const ByteColor& operator()(size_t i, size_t j) const;

 private:
    Array2<ByteColor> _data;
};

typedef std::shared_ptr<ByteImage> ByteImagePtr;

}  // namespace viz
}  // namespace jet

#endif  // INCLUDE_JET_VIZ_IMAGE_H_
