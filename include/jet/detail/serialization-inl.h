// Copyright (c) 2016 Doyub Kim

#ifndef INCLUDE_JET_DETAIL_SERIALIZATION_INL_H_
#define INCLUDE_JET_DETAIL_SERIALIZATION_INL_H_

#include <jet/serialization.h>
#include <cstring>
#include <vector>

namespace jet {

template <typename T>
void serialize(const Array1<T>& array, std::vector<uint8_t>* buffer) {
    size_t size = sizeof(T) * array.size();
    serialize(reinterpret_cast<const uint8_t*>(array.data()), size, buffer);
}

template <typename T>
void deserialize(const std::vector<uint8_t>& buffer, Array1<T>* array) {
    std::vector<uint8_t> data;
    deserialize(buffer, &data);
    array->resize(data.size() / sizeof(T));
    memcpy(array->data(), data.data(), data.size());
}

}  // namespace jet

#endif  // INCLUDE_JET_DETAIL_SERIALIZATION_INL_H_
