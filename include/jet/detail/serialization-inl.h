// Copyright (c) 2018 Doyub Kim
//
// I am making my contributions/submissions to this project solely in my
// personal capacity and am not conveying any rights to any intellectual
// property of any third parties.

#ifndef INCLUDE_JET_DETAIL_SERIALIZATION_INL_H_
#define INCLUDE_JET_DETAIL_SERIALIZATION_INL_H_

#include <jet/serialization.h>

#include <cstring>
#include <vector>

namespace jet {

template <typename T>
void serialize(const ConstArrayAccessor1<T>& array,
               std::vector<uint8_t>* buffer) {
    size_t size = sizeof(T) * array.size();
    serialize(reinterpret_cast<const uint8_t*>(array.data()), size, buffer);
}

template <typename T>
void deserialize(const std::vector<uint8_t>& buffer, Array1<T>* array) {
    std::vector<uint8_t> data;
    deserialize(buffer, &data);
    array->resize(data.size() / sizeof(T));
    memcpy(reinterpret_cast<uint8_t*>(array->data()), data.data(), data.size());
}

}  // namespace jet

#endif  // INCLUDE_JET_DETAIL_SERIALIZATION_INL_H_
