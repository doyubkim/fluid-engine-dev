// Copyright (c) 2016 Doyub Kim

#ifndef INCLUDE_JET_SERIALIZATION_H_
#define INCLUDE_JET_SERIALIZATION_H_

#include <jet/array1.h>
#include <vector>

namespace jet {

//! Abstract base class for any serializable class.
class Serializable {
 public:
    //! Serializes this instance into the flat buffer.
    virtual void serialize(std::vector<uint8_t>* buffer) const = 0;

    //! Deserializes this instance from the flat buffer.
    virtual void deserialize(const std::vector<uint8_t>& buffer) = 0;
};

//! Serializes serializable object.
void serialize(const Serializable* serializable, std::vector<uint8_t>* buffer);

//! Serializes data chunk using common schema.
void serialize(const uint8_t* data, size_t size, std::vector<uint8_t>* buffer);

template <typename T>
void serialize(const Array1<T>& array, std::vector<uint8_t>* buffer);


//! Serializes serializable object.
void deserialize(
    const std::vector<uint8_t>& buffer,
    Serializable* serializable);

//! Serializes data chunk using common schema.
void deserialize(
    const std::vector<uint8_t>& buffer, std::vector<uint8_t>* data);

template <typename T>
void deserialize(const std::vector<uint8_t>& buffer, Array1<T>* array);

}  // namespace jet

#include "detail/serialization-inl.h"

#endif  // INCLUDE_JET_SERIALIZATION_H_
