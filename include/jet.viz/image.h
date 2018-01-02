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

//! Simple 8-bit image representation.
class ByteImage final {
 public:
    //! Constructs an empty image.
    ByteImage();

    //!
    //! \brief Constructs an image with given parameters.
    //!
    //! \param width Width of the new image.
    //! \param height Height of the new image.
    //! \param initialValue Initial color value of the new image.
    //!
    ByteImage(size_t width, size_t height,
              const ByteColor& initialValue = ByteColor());

    //!
    //! \brief Constructs an image from a file.
    //!
    //! \param filename Image file to load.
    //!
    explicit ByteImage(const std::string& filename);

    //! Clears the image and make it zero-sized.
    void clear();

    //!
    //! \brief Resizes the image with given parameters.
    //!
    //! \param width Width of the new image.
    //! \param height Height of the new image.
    //! \param initialValue Initial color value of the new pixels.
    //!
    void resize(size_t width, size_t height,
                const ByteColor& initialValue = ByteColor());

    //! Returns the size of the image.
    Size2 size() const;

    //! Returns mutable raw pointer of the image data.
    ByteColor* data();

    //! Returns immutable raw pointer of the image data.
    const ByteColor* data() const;

    //! Returns mutable pixel reference to (i, j).
    ByteColor& operator()(size_t i, size_t j);

    //! Returns immutable pixel reference to (i, j).
    const ByteColor& operator()(size_t i, size_t j) const;

 private:
    Array2<ByteColor> _data;
};

//! Shared pointer type of ByteImage.
typedef std::shared_ptr<ByteImage> ByteImagePtr;

}  // namespace viz
}  // namespace jet

#endif  // INCLUDE_JET_VIZ_IMAGE_H_
