// Copyright (c) 2018 Doyub Kim
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

namespace jet {

namespace viz {

//! Rendering parameters set.
class RenderParameters final {
 public:
    //! Rendering parameter types.
    enum class Type {
        kInt,
        kUInt,
        kFloat,
        kFloat2,
        kFloat3,
        kFloat4,
        kMatrix,
    };

    //! Rendering parameter metadata.
    struct Metadata {
        size_t offset;
        Type type;
    };

    //! Adds a 32-bit integer parameter.
    void add(const std::string& name, int32_t defaultValue);

    //! Adds a 32-bit unsigned integer parameter.
    void add(const std::string& name, uint32_t defaultValue);

    //! Adds a 32-bit float parameter.
    void add(const std::string& name, float defaultValue);

    //! Adds a 32-bit 2-D float parameter.
    void add(const std::string& name, const Vector2F& defaultValue);

    //! Adds a 32-bit 3-D float parameter.
    void add(const std::string& name, const Vector3F& defaultValue);

    //! Adds a 32-bit 4-D float parameter.
    void add(const std::string& name, const Vector4F& defaultValue);

    //! Adds a 32-bit 4x4 float matrix parameter.
    void add(const std::string& name, const Matrix4x4F& defaultValue);

    //! Sets a 32-bit integer parameter.
    void set(const std::string& name, int32_t value);

    //! Sets a 32-bit unsigned integer parameter.
    void set(const std::string& name, uint32_t value);

    //! Sets a 32-bit float parameter.
    void set(const std::string& name, float value);

    //! Sets a 32-bit 2-D float parameter.
    void set(const std::string& name, const Vector2F& value);

    //! Sets a 32-bit 3-D float parameter.
    void set(const std::string& name, const Vector3F& value);

    //! Sets a 32-bit 4-D float parameter.
    void set(const std::string& name, const Vector4F& value);

    //! Sets a 32-bit 4x4 float matrix parameter.
    void set(const std::string& name, const Matrix4x4F& value);

    //! Returns true if a parameter exists in the set with \p name.
    bool has(const std::string& name) const;

    //! Returns every name of the parameters.
    const std::vector<std::string>& names() const;

    //! Returns raw pointer to the buffer.
    const int32_t* buffer() const;

    //! Returns raw pointer to the parameter in the buffer.
    const int32_t* buffer(const std::string& name) const;

    //! Returns size of the buffer in bytes.
    size_t bufferSizeInBytes() const;

    //! Returns metadata of the parameter with \p name.
    Metadata metadata(const std::string& name) const;

 private:
    size_t _lastParameterOffset = 0;
    std::unordered_map<std::string, Metadata> _metadata;
    std::vector<std::string> _names;
    std::vector<int32_t> _buffer;

    void add(const std::string& name, const int32_t* defaultValue,
             Type elementType);

    void set(const std::string& name, const int32_t* value);
};

}  // namespace viz

}  // namespace jet

#endif  // INCLUDE_JET_VIZ_RENDER_PARAMETERS_H_
