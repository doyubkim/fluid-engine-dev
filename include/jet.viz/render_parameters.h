// Copyright (c) 2017 Doyub Kim
//
// I am making my contributions/submissions to this project solely in my
// personal capacity and am not conveying any rights to any intellectual
// property of any third parties.

#ifndef INCLUDE_JET_VIZ_RENDER_PARAMETERS_H_
#define INCLUDE_JET_VIZ_RENDER_PARAMETERS_H_

#include <jet/matrix4x4.h>
#include <cstdint>
#include <string>
#include <unordered_map>
#include <vector>

namespace jet { namespace viz {

class RenderParameters final {
 public:
    enum class Type {
        kInt,
        kUInt,
        kFloat,
        kFloat2,
        kFloat3,
        kFloat4,
        kMatrix,
    };

    struct Metadata {
        std::size_t offset;
        Type type;
    };

    void add(const std::string& name, std::int32_t defaultValue);

    void add(const std::string& name, std::uint32_t defaultValue);

    void add(const std::string& name, float defaultValue);

    void add(const std::string& name, const Vector2F& defaultValue);

    void add(const std::string& name, const Vector3F& defaultValue);

    void add(const std::string& name, const Vector4F& defaultValue);

    void add(const std::string& name, const Matrix4x4F& defaultValue);

    void set(const std::string& name, std::int32_t value);

    void set(const std::string& name, std::uint32_t value);

    void set(const std::string& name, float value);

    void set(const std::string& name, const Vector2F& value);

    void set(const std::string& name, const Vector3F& value);

    void set(const std::string& name, const Vector4F& value);

    void set(const std::string& name, const Matrix4x4F& value);

    bool has(const std::string& name) const;

    const std::vector<std::string>& names() const;

    const std::int32_t* buffer() const;

    const std::int32_t* buffer(const std::string& name) const;

    std::size_t bufferSizeInBytes() const;

    Metadata metadata(const std::string& name) const;

 private:
    std::size_t _lastParameterOffset = 0;
    std::unordered_map<std::string, Metadata> _metadata;
    std::vector<std::string> _names;
    std::vector<std::int32_t> _buffer;

    void add(const std::string& name, const std::int32_t* defaultValue,
             Type elementType);

    void set(const std::string& name, const std::int32_t* value);
};

} }  // namespace jet::viz

#endif  // INCLUDE_JET_VIZ_RENDER_PARAMETERS_H_
