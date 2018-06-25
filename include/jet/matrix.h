// Copyright (c) 2018 Doyub Kim
//
// I am making my contributions/submissions to this project solely in my
// personal capacity and am not conveying any rights to any intellectual
// property of any third parties.

#ifndef INCLUDE_JET_MATRIX_H_
#define INCLUDE_JET_MATRIX_H_

#include <jet/functors.h>
#include <jet/macros.h>
#include <jet/matrix_expression.h>
#include <jet/matrix_operation.h>

#include <array>
#include <numeric>
#include <tuple>
#include <vector>

namespace jet {

////////////////////////////////////////////////////////////////////////////////
// MARK: Matrix Class (Static)

template <typename T, size_t Rows, size_t Cols>
class Matrix final
    : public MatrixExpression<T, Matrix<T, Rows, Cols>>,
      public MatrixOperation<T, Rows, Cols, Matrix<T, Rows, Cols>> {
 public:
    static_assert(isMatrixSizeStatic<Rows, Cols>(),
                  "This class should be a static-sized matrix.");

    using value_type = T;
    using reference = T&;
    using const_reference = const T&;
    using pointer = T*;
    using const_pointer = const T*;
    using iterator = pointer;
    using const_iterator = const_pointer;
    using MatrixOperation<T, Rows, Cols, Matrix<T, Rows, Cols>>::copyFrom;

    constexpr Matrix() : _elements{} {}

    constexpr Matrix(const_reference value) { fill(value); }

    template <typename... Args>
    constexpr Matrix(const_reference first, Args... rest)
        : _elements{{first, static_cast<value_type>(rest)...}} {}

    template <typename E>
    Matrix(const MatrixExpression<T, E>& expression);

    Matrix(NestedInitializerListsT<T, 2> lst);

    Matrix(const_pointer ptr);

    constexpr Matrix(const Matrix& other) : _elements(other._elements) {}

    void fill(const T& val);

    void fill(const std::function<T(size_t i)>& func);

    void fill(const std::function<T(size_t i, size_t j)>& func);

    void swap(Matrix& other);

    constexpr size_t rows() const;

    constexpr size_t cols() const;

    constexpr iterator begin();

    constexpr const_iterator begin() const;

    constexpr iterator end();

    constexpr const_iterator end() const;

    constexpr pointer data();

    constexpr const_pointer data() const;

    reference operator[](size_t i);

    const_reference operator[](size_t i) const;

 private:
    std::array<T, Rows * Cols> _elements;
};

// MARK: Specialized Matrix for 1, 2, 3, and 4-D Vector Types

template <typename T>
class Matrix<T, 1, 1> final : public MatrixExpression<T, Matrix<T, 1, 1>>,
                              public MatrixOperation<T, 1, 1, Matrix<T, 1, 1>> {
 public:
    using value_type = T;
    using reference = T&;
    using const_reference = const T&;
    using pointer = T*;
    using const_pointer = const T*;
    using iterator = pointer;
    using const_iterator = const_pointer;

    value_type x;

    constexpr Matrix() : x(T{}) {}

    constexpr Matrix(const T& x_) : x(x_) {}

    constexpr Matrix(const Matrix& other) : x(other.x) {}

    template <typename E>
    Matrix(const MatrixExpression<T, E>& expression);

    // Simple setters/modifiers

    void fill(const T& val);

    void fill(const std::function<T(size_t i)>& func);

    void fill(const std::function<T(size_t i, size_t j)>& func);

    void swap(Matrix& other);

    // Simple getters

    constexpr size_t rows() const;

    constexpr size_t cols() const;

    constexpr iterator begin();

    constexpr const_iterator begin() const;

    constexpr iterator end();

    constexpr const_iterator end() const;

    constexpr pointer data();

    constexpr const_pointer data() const;

    // Operator overloadings

    reference operator[](size_t i);

    const_reference operator[](size_t i) const;
};

template <typename T>
class Matrix<T, 2, 1> final : public MatrixExpression<T, Matrix<T, 2, 1>>,
                              public MatrixOperation<T, 2, 1, Matrix<T, 2, 1>> {
 public:
    using value_type = T;
    using reference = T&;
    using const_reference = const T&;
    using pointer = T*;
    using const_pointer = const T*;
    using iterator = pointer;
    using const_iterator = const_pointer;

    value_type x;
    value_type y;

    constexpr Matrix() : x(T{}), y(T{}) {}

    constexpr Matrix(const T& x_, const T& y_) : x(x_), y(y_) {}

    constexpr Matrix(const Matrix& other) : x(other.x), y(other.y) {}

    template <typename E>
    Matrix(const MatrixExpression<T, E>& expression);

    // Simple setters/modifiers

    void fill(const T& val);

    void fill(const std::function<T(size_t i)>& func);

    void fill(const std::function<T(size_t i, size_t j)>& func);

    void swap(Matrix& other);

    // Simple getters

    constexpr size_t rows() const;

    constexpr size_t cols() const;

    constexpr iterator begin();

    constexpr const_iterator begin() const;

    constexpr iterator end();

    constexpr const_iterator end() const;

    constexpr pointer data();

    constexpr const_pointer data() const;

    //! Returns the reflection vector to the surface with given surface normal.
    constexpr Matrix reflected(const Matrix& normal) const;

    //! Returns the projected vector to the surface with given surface normal.
    constexpr Matrix projected(const Matrix& normal) const;

    //! Returns the tangential vector for this vector.
    constexpr Matrix tangential() const;

    // Binary operators

    template <typename E>
    constexpr value_type cross(const MatrixExpression<T, E>& expression) const;

    // Operator overloadings

    reference operator[](size_t i);

    const_reference operator[](size_t i) const;
};

template <typename T>
class Matrix<T, 3, 1> final : public MatrixExpression<T, Matrix<T, 3, 1>>,
                              public MatrixOperation<T, 3, 1, Matrix<T, 3, 1>> {
 public:
    using value_type = T;
    using reference = T&;
    using const_reference = const T&;
    using pointer = T*;
    using const_pointer = const T*;
    using iterator = pointer;
    using const_iterator = const_pointer;

    value_type x;
    value_type y;
    value_type z;

    constexpr Matrix() : x(T{}), y(T{}), z(T{}) {}

    constexpr Matrix(const Matrix<T, 2, 1>& xy_, const T& z_)
        : x(xy_.x), y(xy_.y), z(z_) {}

    constexpr Matrix(const T& x_, const T& y_, const T& z_)
        : x(x_), y(y_), z(z_) {}

    constexpr Matrix(const Matrix& other)
        : x(other.x), y(other.y), z(other.z) {}

    template <typename E>
    Matrix(const MatrixExpression<T, E>& expression);

    // Simple setters/modifiers

    void fill(const T& val);

    void fill(const std::function<T(size_t i)>& func);

    void fill(const std::function<T(size_t i, size_t j)>& func);

    void swap(Matrix& other);

    // Simple getters

    constexpr size_t rows() const;

    constexpr size_t cols() const;

    constexpr iterator begin();

    constexpr const_iterator begin() const;

    constexpr iterator end();

    constexpr const_iterator end() const;

    constexpr pointer data();

    constexpr const_pointer data() const;

    //! Returns the reflection vector to the surface with given surface normal.
    constexpr Matrix reflected(const Matrix& normal) const;

    //! Returns the projected vector to the surface with given surface normal.
    constexpr Matrix projected(const Matrix& normal) const;

    //! Returns the tangential vector for this vector.
    std::tuple<Matrix, Matrix> tangential() const;

    // Binary operators

    template <typename E>
    constexpr Matrix cross(const MatrixExpression<T, E>& expression) const;

    // Operator overloadings

    reference operator[](size_t i);

    const_reference operator[](size_t i) const;
};

template <typename T>
class Matrix<T, 4, 1> final : public MatrixExpression<T, Matrix<T, 4, 1>>,
                              public MatrixOperation<T, 4, 1, Matrix<T, 4, 1>> {
 public:
    using value_type = T;
    using reference = T&;
    using const_reference = const T&;
    using pointer = T*;
    using const_pointer = const T*;
    using iterator = pointer;
    using const_iterator = const_pointer;

    value_type x;
    value_type y;
    value_type z;
    value_type w;

    constexpr Matrix() : x(T{}), y(T{}), z(T{}), w(T{}) {}

    constexpr Matrix(const T& x_, const T& y_, const T& z_, const T& w_)
        : x(x_), y(y_), z(z_), w(w_) {}

    constexpr Matrix(const Matrix& other)
        : x(other.x), y(other.y), z(other.z), w(other.w) {}

    template <typename E>
    Matrix(const MatrixExpression<T, E>& expression);

    // Simple setters/modifiers

    void fill(const T& val);

    void fill(const std::function<T(size_t i)>& func);

    void fill(const std::function<T(size_t i, size_t j)>& func);

    void swap(Matrix& other);

    // Simple getters

    constexpr size_t rows() const;

    constexpr size_t cols() const;

    constexpr iterator begin();

    constexpr const_iterator begin() const;

    constexpr iterator end();

    constexpr const_iterator end() const;

    constexpr pointer data();

    constexpr const_pointer data() const;

    // Operator overloadings

    reference operator[](size_t i);

    const_reference operator[](size_t i) const;
};

////////////////////////////////////////////////////////////////////////////////
// MARK: Matrix Class (Dynamic)

template <typename T>
class Matrix<T, kMatrixSizeDynamic, kMatrixSizeDynamic> final
    : public MatrixExpression<
          T, Matrix<T, kMatrixSizeDynamic, kMatrixSizeDynamic>>,
      public MatrixOperation<
          T, kMatrixSizeDynamic, kMatrixSizeDynamic,
          Matrix<T, kMatrixSizeDynamic, kMatrixSizeDynamic>> {
 public:
    using value_type = T;
    using reference = T&;
    using const_reference = const T&;
    using pointer = T*;
    using const_pointer = const T*;
    using iterator = pointer;
    using const_iterator = const_pointer;
    using MatrixOperation<
        T, kMatrixSizeDynamic, kMatrixSizeDynamic,
        Matrix<T, kMatrixSizeDynamic, kMatrixSizeDynamic>>::copyFrom;

    Matrix() {}

    Matrix(size_t rows, size_t cols, const_reference value = value_type{}) {
        _elements.resize(rows * cols);
        _rows = rows;
        _cols = cols;
        fill(value);
    }

    // template <typename... Args>
    // Matrix(const_reference first, Args... rest)
    //     : _elements{{first, static_cast<value_type>(rest)...}} {}

    template <typename E>
    Matrix(const MatrixExpression<T, E>& expression)
        : Matrix(expression.rows(), expression.cols()) {
        copyFrom(expression);
    }

    Matrix(NestedInitializerListsT<T, 2> lst) {
        size_t i = 0;
        for (auto rows : lst) {
            size_t j = 0;
            for (auto col : rows) {
                ++j;
            }
            _cols = j;
            ++i;
        }
        _rows = i;
        _elements.resize(_rows * _cols);

        i = 0;
        for (auto rows : lst) {
            JET_ASSERT(i < _rows);
            size_t j = 0;
            for (auto col : rows) {
                JET_ASSERT(j < _cols);
                (*this)(i, j) = col;
                ++j;
            }
            ++i;
        }
    }

    explicit Matrix(size_t rows, size_t cols, const_pointer ptr)
        : Matrix(rows, cols) {
        size_t cnt = 0;
        for (size_t i = 0; i < rows; ++i) {
            for (size_t j = 0; j < cols; ++j) {
                (*this)(i, j) = ptr[cnt++];
            }
        }
    }

    Matrix(const Matrix& other)
        : _elements(other._elements), _rows(other._rows), _cols(other._cols) {}

    Matrix(Matrix&& other) { *this = std::move(other); }

    void fill(const T& val) {
        std::fill(_elements.begin(), _elements.end(), val);
    }

    void fill(const std::function<T(size_t i)>& func) {
        for (size_t i = 0; i < _elements.size(); ++i) {
            _elements[i] = func(i);
        }
    }

    void fill(const std::function<T(size_t i, size_t j)>& func) {
        for (size_t i = 0; i < rows(); ++i) {
            for (size_t j = 0; j < cols(); ++j) {
                (*this)(i, j) = func(i, j);
            }
        }
    }

    void swap(Matrix& other) {
        _elements.swap(other._elements);
        std::swap(_rows, other._rows);
        std::swap(_cols, other._cols);
    }

    void resize(size_t rows, size_t cols, const_reference val = value_type{}) {
        Matrix newMatrix{rows, cols, val};
        size_t minRows = std::min(rows, _rows);
        size_t minCols = std::min(cols, _cols);
        for (size_t i = 0; i < minRows; ++i) {
            for (size_t j = 0; j < minCols; ++j) {
                newMatrix(i, j) = (*this)(i, j);
            }
        }
        *this = std::move(newMatrix);
    }

    size_t rows() const { return _rows; }

    size_t cols() const { return _cols; }

    iterator begin() { return &_elements[0]; }

    const_iterator begin() const { return &_elements[0]; }

    iterator end() { return begin() + _rows * _cols; }

    const_iterator end() const { return begin() + _rows * _cols; }

    pointer data() { return &_elements[0]; }

    const_pointer data() const { return &_elements[0]; }

    reference operator[](size_t i) {
        JET_ASSERT(i < _rows * _Cols);
        return _elements[i];
    }

    const_reference operator[](size_t i) const {
        JET_ASSERT(i < _rows * _Cols);
        return _elements[i];
    }

    Matrix& operator=(const Matrix& other) {
        _elements = other._elements;
        _rows = other._rows;
        _cols = other._cols;
        return *this;
    }

    Matrix& operator=(Matrix&& other) {
        _elements = std::move(other._elements);
        _rows = other._rows;
        _cols = other._cols;
        other._rows = 0;
        other._cols = 0;
        return *this;
    }

 private:
    std::vector<T> _elements;
    size_t _rows = 0;
    size_t _cols = 0;
};

// MARK: Specialized Matrix for Dynamic Vector Type

template <typename T>
class Matrix<T, kMatrixSizeDynamic, 1> final
    : public MatrixExpression<T, Matrix<T, kMatrixSizeDynamic, 1>>,
      public MatrixOperation<T, kMatrixSizeDynamic, 1,
                             Matrix<T, kMatrixSizeDynamic, 1>> {
 public:
    using value_type = T;
    using reference = T&;
    using const_reference = const T&;
    using pointer = T*;
    using const_pointer = const T*;
    using iterator = pointer;
    using const_iterator = const_pointer;
    using MatrixOperation<T, kMatrixSizeDynamic, 1,
                          Matrix<T, kMatrixSizeDynamic, 1>>::copyFrom;

    Matrix() {}

    Matrix(size_t rows, const_reference value = value_type{}) {
        _elements.resize(rows);
        fill(value);
    }

    // template <typename... Args>
    // Matrix(const_reference first, Args... rest)
    //     : _elements{{first, static_cast<value_type>(rest)...}} {}

    template <typename E>
    Matrix(const MatrixExpression<T, E>& expression)
        : Matrix(expression.rows(), expression.cols()) {
        copyFrom(expression);
    }

    Matrix(NestedInitializerListsT<T, 1> lst) {
        size_t sz = lst.size();
        _elements.resize(sz);

        size_t i = 0;
        for (auto row : lst) {
            _elements[i] = row;
            ++i;
        }
    }

    explicit Matrix(size_t rows, const_pointer ptr) : Matrix(rows) {
        size_t cnt = 0;
        for (size_t i = 0; i < rows; ++i) {
            (*this)[i] = ptr[cnt++];
        }
    }

    Matrix(const Matrix& other) : _elements(other._elements) {}

    Matrix(Matrix&& other) { *this = std::move(other); }

    void fill(const T& val) {
        std::fill(_elements.begin(), _elements.end(), val);
    }

    void fill(const std::function<T(size_t i)>& func) {
        for (size_t i = 0; i < _elements.size(); ++i) {
            _elements[i] = func(i);
        }
    }

    void fill(const std::function<T(size_t i, size_t j)>& func) {
        for (size_t i = 0; i < rows(); ++i) {
            _elements[i] = func(i, 0);
        }
    }

    void swap(Matrix& other) { _elements.swap(other._elements); }

    void resize(size_t rows, const_reference val = value_type{}) {
        _elements.resize(rows, val);
    }

    size_t rows() const { return _elements.size(); }

    constexpr size_t cols() const { return 1; }

    iterator begin() { return &_elements[0]; }

    const_iterator begin() const { return &_elements[0]; }

    iterator end() { return begin() + _elements.size(); }

    const_iterator end() const { return begin() + _elements.size(); }

    pointer data() { return &_elements[0]; }

    const_pointer data() const { return &_elements[0]; }

    reference operator[](size_t i) {
        JET_ASSERT(i < _elements.size());
        return _elements[i];
    }

    const_reference operator[](size_t i) const {
        JET_ASSERT(i < _elements.size());
        return _elements[i];
    }

    Matrix& operator=(const Matrix& other) {
        _elements = other._elements;
        return *this;
    }

    Matrix& operator=(Matrix&& other) {
        _elements = std::move(other._elements);
        return *this;
    }

 private:
    std::vector<T> _elements;
};

////////////////////////////////////////////////////////////////////////////////
// MARK: Type Aliasing

template <typename T>
using Matrix2x2 = Matrix<T, 2, 2>;

template <typename T>
using Matrix3x3 = Matrix<T, 3, 3>;

template <typename T>
using Matrix4x4 = Matrix<T, 4, 4>;

using Matrix2x2B = Matrix2x2<int8_t>;
using Matrix2x2UB = Matrix2x2<uint8_t>;
using Matrix2x2S = Matrix2x2<int16_t>;
using Matrix2x2US = Matrix2x2<uint16_t>;
using Matrix2x2I = Matrix2x2<int32_t>;
using Matrix2x2UI = Matrix2x2<uint32_t>;
using Matrix2x2L = Matrix2x2<int64_t>;
using Matrix2x2UL = Matrix2x2<uint64_t>;
using Matrix2x2F = Matrix2x2<float>;
using Matrix2x2D = Matrix2x2<double>;
using Matrix2x2Z = Matrix2x2<ssize_t>;
using Matrix2x2UZ = Matrix2x2<size_t>;

using Matrix3x3B = Matrix3x3<int8_t>;
using Matrix3x3UB = Matrix3x3<uint8_t>;
using Matrix3x3S = Matrix3x3<int16_t>;
using Matrix3x3US = Matrix3x3<uint16_t>;
using Matrix3x3I = Matrix3x3<int32_t>;
using Matrix3x3UI = Matrix3x3<uint32_t>;
using Matrix3x3L = Matrix3x3<int64_t>;
using Matrix3x3UL = Matrix3x3<uint64_t>;
using Matrix3x3F = Matrix3x3<float>;
using Matrix3x3D = Matrix3x3<double>;
using Matrix3x3Z = Matrix3x3<ssize_t>;
using Matrix3x3UZ = Matrix3x3<size_t>;

using Matrix4x4B = Matrix4x4<int8_t>;
using Matrix4x4UB = Matrix4x4<uint8_t>;
using Matrix4x4S = Matrix4x4<int16_t>;
using Matrix4x4US = Matrix4x4<uint16_t>;
using Matrix4x4I = Matrix4x4<int32_t>;
using Matrix4x4UI = Matrix4x4<uint32_t>;
using Matrix4x4L = Matrix4x4<int64_t>;
using Matrix4x4UL = Matrix4x4<uint64_t>;
using Matrix4x4F = Matrix4x4<float>;
using Matrix4x4D = Matrix4x4<double>;
using Matrix4x4Z = Matrix4x4<ssize_t>;
using Matrix4x4UZ = Matrix4x4<size_t>;

template <typename T, size_t Rows>
using Vector = Matrix<T, Rows, 1>;

template <typename T>
using Vector1 = Vector<T, 1>;

template <typename T>
using Vector2 = Vector<T, 2>;

template <typename T>
using Vector3 = Vector<T, 3>;

template <typename T>
using Vector4 = Vector<T, 4>;

using Vector1B = Vector1<int8_t>;
using Vector1UB = Vector1<uint8_t>;
using Vector1S = Vector1<int16_t>;
using Vector1US = Vector1<uint16_t>;
using Vector1I = Vector1<int32_t>;
using Vector1UI = Vector1<uint32_t>;
using Vector1L = Vector1<int64_t>;
using Vector1UL = Vector1<uint64_t>;
using Vector1F = Vector1<float>;
using Vector1D = Vector1<double>;
using Vector1Z = Vector1<ssize_t>;
using Vector1UZ = Vector1<size_t>;

using Vector2B = Vector2<int8_t>;
using Vector2UB = Vector2<uint8_t>;
using Vector2S = Vector2<int16_t>;
using Vector2US = Vector2<uint16_t>;
using Vector2I = Vector2<int32_t>;
using Vector2UI = Vector2<uint32_t>;
using Vector2L = Vector2<int64_t>;
using Vector2UL = Vector2<uint64_t>;
using Vector2F = Vector2<float>;
using Vector2D = Vector2<double>;
using Vector2Z = Vector2<ssize_t>;
using Vector2UZ = Vector2<size_t>;

using Vector3B = Vector3<int8_t>;
using Vector3UB = Vector3<uint8_t>;
using Vector3S = Vector3<int16_t>;
using Vector3US = Vector3<uint16_t>;
using Vector3I = Vector3<int32_t>;
using Vector3UI = Vector3<uint32_t>;
using Vector3L = Vector3<int64_t>;
using Vector3UL = Vector3<uint64_t>;
using Vector3F = Vector3<float>;
using Vector3D = Vector3<double>;
using Vector3Z = Vector3<ssize_t>;
using Vector3UZ = Vector3<size_t>;

using Vector4B = Vector4<int8_t>;
using Vector4UB = Vector4<uint8_t>;
using Vector4S = Vector4<int16_t>;
using Vector4US = Vector4<uint16_t>;
using Vector4I = Vector4<int32_t>;
using Vector4UI = Vector4<uint32_t>;
using Vector4L = Vector4<int64_t>;
using Vector4UL = Vector4<uint64_t>;
using Vector4F = Vector4<float>;
using Vector4D = Vector4<double>;
using Vector4Z = Vector4<ssize_t>;
using Vector4UZ = Vector4<size_t>;

template <typename T>
using MatrixMxN = Matrix<T, kMatrixSizeDynamic, kMatrixSizeDynamic>;

using MatrixMxNB = MatrixMxN<int8_t>;
using MatrixMxNUB = MatrixMxN<uint8_t>;
using MatrixMxNS = MatrixMxN<int16_t>;
using MatrixMxNUS = MatrixMxN<uint16_t>;
using MatrixMxNI = MatrixMxN<int32_t>;
using MatrixMxNUI = MatrixMxN<uint32_t>;
using MatrixMxNL = MatrixMxN<int64_t>;
using MatrixMxNUL = MatrixMxN<uint64_t>;
using MatrixMxNF = MatrixMxN<float>;
using MatrixMxND = MatrixMxN<double>;
using MatrixMxNZ = MatrixMxN<ssize_t>;
using MatrixMxNUZ = MatrixMxN<size_t>;

template <typename T>
using VectorN = Matrix<T, kMatrixSizeDynamic, 1>;

using VectorNB = VectorN<int8_t>;
using VectorNUB = VectorN<uint8_t>;
using VectorNS = VectorN<int16_t>;
using VectorNUS = VectorN<uint16_t>;
using VectorNI = VectorN<int32_t>;
using VectorNUI = VectorN<uint32_t>;
using VectorNL = VectorN<int64_t>;
using VectorNUL = VectorN<uint64_t>;
using VectorNF = VectorN<float>;
using VectorND = VectorN<double>;
using VectorNZ = VectorN<ssize_t>;
using VectorNUZ = VectorN<size_t>;

////////////////////////////////////////////////////////////////////////////////
// MARK: Matrix Operators

// MARK: Binary Operators

// *

template <typename T, size_t Rows>
[[deprecated("Use elemMul instead")]] constexpr auto operator*(
    const Vector<T, Rows>& a, const Vector<T, Rows>& b);

// /

template <typename T, size_t Rows>
[[deprecated("Use elemDiv instead")]] constexpr auto operator/(
    const Vector<T, Rows>& a, const Vector<T, Rows>& b);

// MARK: Assignment Operators

// +=

template <typename T, size_t Rows, size_t Cols, typename M2>
void operator+=(Matrix<T, Rows, Cols>& a, const MatrixExpression<T, M2>& b);

template <typename T, size_t Rows, size_t Cols>
void operator+=(Matrix<T, Rows, Cols>& a, const T& b);

// -=

template <typename T, size_t Rows, size_t Cols, typename M2>
void operator-=(Matrix<T, Rows, Cols>& a, const MatrixExpression<T, M2>& b);

template <typename T, size_t Rows, size_t Cols>
void operator-=(Matrix<T, Rows, Cols>& a, const T& b);

// *=

template <typename T, size_t Rows, size_t Cols, typename M2>
void operator*=(Matrix<T, Rows, Cols>& a, const MatrixExpression<T, M2>& b);

template <typename T, size_t Rows, typename M2>
[[deprecated("Use elemIMul instead")]] void operator*=(
    Matrix<T, Rows, 1>& a, const MatrixExpression<T, M2>& b);

template <typename T, size_t Rows, size_t Cols, typename M2>
void elemIMul(Matrix<T, Rows, Cols>& a, const MatrixExpression<T, M2>& b);

template <typename T, size_t Rows, size_t Cols>
void operator*=(Matrix<T, Rows, Cols>& a, const T& b);

// /=

template <typename T, size_t Rows, size_t Cols, typename M2>
[[deprecated("Use elemIDiv instead")]] void operator/=(
    Matrix<T, Rows, Cols>& a, const MatrixExpression<T, M2>& b);

template <typename T, size_t Rows, size_t Cols, typename M2>
void elemIDiv(Matrix<T, Rows, Cols>& a, const MatrixExpression<T, M2>& b);

template <typename T, size_t Rows, size_t Cols>
void operator/=(Matrix<T, Rows, Cols>& a, const T& b);

// MARK: Comparison Operators

template <typename T, size_t Rows, size_t Cols>
constexpr std::enable_if_t<isMatrixSizeStatic<Rows, Cols>(), bool> operator==(
    const Matrix<T, Rows, Cols>& a, const Matrix<T, Rows, Cols>& b);

template <typename T, size_t Rows, size_t Cols, typename E>
bool operator==(const Matrix<T, Rows, Cols>& a,
                const MatrixExpression<T, E>& b);

template <typename T, size_t Rows, size_t Cols, typename E>
bool operator!=(const Matrix<T, Rows, Cols>& a,
                const MatrixExpression<T, E>& b);

// MARK: Simple Utilities

// Static Accumulate

template <typename T, size_t Rows, size_t Cols, typename BinaryOperation>
constexpr std::enable_if_t<isMatrixSizeStatic<Rows, Cols>(), T> accumulate(
    const Matrix<T, Rows, Cols>& a, const T& init, BinaryOperation op);

template <typename T, size_t Rows, size_t Cols>
constexpr std::enable_if_t<isMatrixSizeStatic<Rows, Cols>(), T> accumulate(
    const Matrix<T, Rows, Cols>& a, const T& init);

template <typename T, size_t Rows, size_t Cols>
constexpr std::enable_if_t<isMatrixSizeStatic<Rows, Cols>(), T> accumulate(
    const Matrix<T, Rows, Cols>& a);

// Dynamic Accumulate

template <typename T, size_t Rows, size_t Cols, typename BinaryOperation>
constexpr std::enable_if_t<isMatrixSizeDynamic<Rows, Cols>(), T> accumulate(
    const Matrix<T, Rows, Cols>& a, const T& init, BinaryOperation op);

template <typename T, size_t Rows, size_t Cols>
constexpr std::enable_if_t<isMatrixSizeDynamic<Rows, Cols>(), T> accumulate(
    const Matrix<T, Rows, Cols>& a, const T& init);

template <typename T, size_t Rows, size_t Cols>
constexpr std::enable_if_t<isMatrixSizeDynamic<Rows, Cols>(), T> accumulate(
    const Matrix<T, Rows, Cols>& a);

// Product

template <typename T, size_t Rows, size_t Cols>
constexpr T product(const Matrix<T, Rows, Cols>& a, const T& init);

}  // namespace jet

#include <jet/detail/matrix-inl.h>

#endif  // INCLUDE_JET_MATRIX_H_
