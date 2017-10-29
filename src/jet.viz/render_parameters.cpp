// Copyright (c) 2017 Doyub Kim
//
// I am making my contributions/submissions to this project solely in my
// personal capacity and am not conveying any rights to any intellectual
// property of any third parties.

#include <pch.h>

#include <jet.viz/render_parameters.h>

using namespace jet;
using namespace viz;

inline size_t sizeWithPadding(size_t size) {
    if (size == 0) {
        return 0;
    } else {
        return (((size - 1) / 16) + 1) * 16;
    }
}

inline size_t getNumberOfElements(RenderParameters::Type type) {
    switch (type) {
        case RenderParameters::Type::kInt:
        case RenderParameters::Type::kFloat:
            return 1;
        case RenderParameters::Type::kFloat2:
            return 2;
        case RenderParameters::Type::kFloat3:
            return 3;
        case RenderParameters::Type::kFloat4:
            return 4;
        case RenderParameters::Type::kMatrix:
            return 16;
        default:
            assert(false);
    }

    return 0;
}

void RenderParameters::add(const std::string& name, int32_t defaultValue) {
    add(name, &defaultValue, Type::kInt);
}

void RenderParameters::add(const std::string& name,
                           uint32_t defaultValue) {
    add(name, reinterpret_cast<const int32_t*>(&defaultValue),
        Type::kUInt);
}

void RenderParameters::add(const std::string& name, float defaultValue) {
    add(name, reinterpret_cast<const int32_t*>(&defaultValue),
        Type::kFloat);
}

void RenderParameters::add(const std::string& name,
                           const Vector2F& defaultValue) {
    add(name, reinterpret_cast<const int32_t*>(&defaultValue[0]),
        Type::kFloat2);
}

void RenderParameters::add(const std::string& name,
                           const Vector3F& defaultValue) {
    add(name, reinterpret_cast<const int32_t*>(&defaultValue[0]),
        Type::kFloat3);
}

void RenderParameters::add(const std::string& name,
                           const Vector4F& defaultValue) {
    add(name, reinterpret_cast<const int32_t*>(&defaultValue[0]),
        Type::kFloat4);
}

void RenderParameters::add(const std::string& name,
                           const Matrix4x4F& defaultValue) {
    add(name, reinterpret_cast<const int32_t*>(defaultValue.data()),
        Type::kMatrix);
}

void RenderParameters::set(const std::string& name, int32_t value) {
    set(name, &value);
}

void RenderParameters::set(const std::string& name, uint32_t value) {
    set(name, reinterpret_cast<const int32_t*>(&value));
}

void RenderParameters::set(const std::string& name, float value) {
    set(name, reinterpret_cast<const int32_t*>(&value));
}

void RenderParameters::set(const std::string& name, const Vector2F& value) {
    set(name, reinterpret_cast<const int32_t*>(&value[0]));
}

void RenderParameters::set(const std::string& name, const Vector3F& value) {
    set(name, reinterpret_cast<const int32_t*>(&value[0]));
}

void RenderParameters::set(const std::string& name, const Vector4F& value) {
    set(name, reinterpret_cast<const int32_t*>(&value[0]));
}

void RenderParameters::set(const std::string& name, const Matrix4x4F& value) {
    set(name, reinterpret_cast<const int32_t*>(value.data()));
}

bool RenderParameters::has(const std::string& name) const {
    auto iter = _metadata.find(name);
    return (iter != _metadata.end());
}

const std::vector<std::string>& RenderParameters::names() const {
    return _names;
}

const int32_t* RenderParameters::buffer() const { return _buffer.data(); }

const int32_t* RenderParameters::buffer(const std::string& name) const {
    auto iter = _metadata.find(name);

    if (iter != _metadata.end()) {
        return _buffer.data() + iter->second.offset;
    } else {
        return nullptr;
    }
}

size_t RenderParameters::bufferSizeInBytes() const {
    return _buffer.size() * sizeof(int32_t);
}

RenderParameters::Metadata RenderParameters::metadata(
    const std::string& name) const {
    auto iter = _metadata.find(name);

    if (iter != _metadata.end()) {
        return iter->second;
    } else {
        return Metadata();
    }
}

void RenderParameters::add(const std::string& name,
                           const int32_t* defaultValue, Type type) {
    // Can't add with exiting name
    assert(_metadata.find(name) == _metadata.end());

    Metadata metadata = {_lastParameterOffset, type};
    _metadata[name] = metadata;
    _names.push_back(name);

    size_t numberOfElements = getNumberOfElements(type);

    _buffer.resize(sizeWithPadding(_lastParameterOffset + numberOfElements));
    for (size_t i = 0; i < numberOfElements; ++i) {
        _buffer[_lastParameterOffset + i] = defaultValue[i];
    }

    _lastParameterOffset += numberOfElements;
}

void RenderParameters::set(const std::string& name, const int32_t* value) {
    auto iter = _metadata.find(name);
    assert(iter != _metadata.end());

    size_t offset = iter->second.offset;
    size_t numberOfElements = getNumberOfElements(iter->second.type);

    for (size_t i = 0; i < numberOfElements; ++i) {
        _buffer[offset + i] = value[i];
    }
}
