// Copyright (c) 2016 Doyub Kim

#ifndef INCLUDE_JET_SIZE_H_
#define INCLUDE_JET_SIZE_H_

#include <jet/macros.h>
#include <array>
#include <type_traits>

namespace jet {

//!
//! \brief Generic N-D Size class.
//! \tparam N - Dimension.
//!
template <size_t N>
class Size final {
 public:
    static_assert(
        N > 0, "Size of static-sized Size should be greater than zero.");

    std::array<size_t, N> elements;

    Size();
    template <typename... Params>
    explicit Size(Params... params);
    explicit Size(const std::initializer_list<size_t>& lst);
    Size(const Size& other);

    const size_t& operator[](size_t i) const;
    size_t& operator[](size_t);

 private:
    template <typename... Params>
    void setAt(size_t i, size_t v, Params... params);

    void setAt(size_t i, size_t v);
};

}  // namespace jet

#include "detail/size-inl.h"

#endif  // INCLUDE_JET_SIZE_H_

