// Copyright (c) 2018 Doyub Kim
//
// I am making my contributions/submissions to this project solely in my
// personal capacity and am not conveying any rights to any intellectual
// property of any third parties.

#ifndef INCLUDE_JET_DETAIL_MATRIX_CSR_INL_H_
#define INCLUDE_JET_DETAIL_MATRIX_CSR_INL_H_

#include <jet/cpp_utils.h>
#include <jet/math_utils.h>
#include <jet/matrix_csr.h>
#include <jet/parallel.h>

#include <numeric>

namespace jet {

template <typename T, typename VE>
MatrixCsrVectorMul<T, VE>::MatrixCsrVectorMul(const MatrixCsr<T>& m,
                                              const VE& v)
    : _m(m), _v(v) {
    JET_ASSERT(_m.cols() == _v.size());
}

template <typename T, typename VE>
MatrixCsrVectorMul<T, VE>::MatrixCsrVectorMul(const MatrixCsrVectorMul& other)
    : _m(other._m), _v(other._v) {}

template <typename T, typename VE>
size_t MatrixCsrVectorMul<T, VE>::size() const {
    return _m.rows();
}

template <typename T, typename VE>
T MatrixCsrVectorMul<T, VE>::operator[](size_t i) const {
    auto rp = _m.rowPointersBegin();
    auto ci = _m.columnIndicesBegin();
    auto nnz = _m.nonZeroBegin();

    size_t colBegin = rp[i];
    size_t colEnd = rp[i + 1];

    T sum = 0;

    for (size_t jj = colBegin; jj < colEnd; ++jj) {
        size_t j = ci[jj];
        sum += nnz[jj] * _v[j];
    }

    return sum;
}

//

template <typename T, typename ME>
MatrixCsrMatrixMul<T, ME>::MatrixCsrMatrixMul(const MatrixCsr<T>& m1,
                                              const ME& m2)
    : _m1(m1),
      _m2(m2),
      _nnz(m1.nonZeroData()),
      _rp(m1.rowPointersData()),
      _ci(m1.columnIndicesData()) {}

template <typename T, typename ME>
Size2 MatrixCsrMatrixMul<T, ME>::size() const {
    return {rows(), cols()};
}

template <typename T, typename ME>
size_t MatrixCsrMatrixMul<T, ME>::rows() const {
    return _m1.rows();
}

template <typename T, typename ME>
size_t MatrixCsrMatrixMul<T, ME>::cols() const {
    return _m2.cols();
}

template <typename T, typename ME>
T MatrixCsrMatrixMul<T, ME>::operator()(size_t i, size_t j) const {
    size_t colBegin = _rp[i];
    size_t colEnd = _rp[i + 1];

    T sum = 0;
    for (size_t kk = colBegin; kk < colEnd; ++kk) {
        size_t k = _ci[kk];
        sum += _nnz[kk] * _m2(k, j);
    }

    return sum;
}

//

template <typename T>
MatrixCsr<T>::Element::Element() : i(0), j(0), value(0) {}

template <typename T>
MatrixCsr<T>::Element::Element(size_t i_, size_t j_, const T& value_)
    : i(i_), j(j_), value(value_) {}

//

template <typename T>
MatrixCsr<T>::MatrixCsr() {
    clear();
}

template <typename T>
MatrixCsr<T>::MatrixCsr(
    const std::initializer_list<std::initializer_list<T>>& lst, T epsilon) {
    compress(lst, epsilon);
}

template <typename T>
template <typename E>
MatrixCsr<T>::MatrixCsr(const MatrixExpression<T, E>& other, T epsilon) {
    compress(other, epsilon);
}

template <typename T>
MatrixCsr<T>::MatrixCsr(const MatrixCsr& other) {
    set(other);
}

template <typename T>
MatrixCsr<T>::MatrixCsr(MatrixCsr&& other) {
    (*this) = std::move(other);
}

template <typename T>
void MatrixCsr<T>::clear() {
    _size = {0, 0};
    _nonZeros.clear();
    _rowPointers.clear();
    _columnIndices.clear();
    _rowPointers.push_back(0);
}

template <typename T>
void MatrixCsr<T>::set(const T& s) {
    std::fill(_nonZeros.begin(), _nonZeros.end(), s);
}

template <typename T>
void MatrixCsr<T>::set(const MatrixCsr& other) {
    _size = other._size;
    _nonZeros = other._nonZeros;
    _rowPointers = other._rowPointers;
    _columnIndices = other._columnIndices;
}

template <typename T>
void MatrixCsr<T>::reserve(size_t rows, size_t cols, size_t numNonZeros) {
    _size = Size2(rows, cols);
    _rowPointers.resize(_size.x + 1);
    _nonZeros.resize(numNonZeros);
    _columnIndices.resize(numNonZeros);
}

template <typename T>
void MatrixCsr<T>::compress(
    const std::initializer_list<std::initializer_list<T>>& lst, T epsilon) {
    size_t numRows = lst.size();
    size_t numCols = (numRows > 0) ? lst.begin()->size() : 0;

    _size = {numRows, numCols};
    _nonZeros.clear();
    _rowPointers.clear();
    _columnIndices.clear();

    auto rowIter = lst.begin();
    for (size_t i = 0; i < numRows; ++i) {
        JET_ASSERT(numCols == rowIter->size());
        _rowPointers.push_back(_nonZeros.size());

        auto colIter = rowIter->begin();
        for (size_t j = 0; j < numCols; ++j) {
            if (std::fabs(*colIter) > epsilon) {
                _nonZeros.push_back(*colIter);
                _columnIndices.push_back(j);
            }

            ++colIter;
        }
        ++rowIter;
    }

    _rowPointers.push_back(_nonZeros.size());
}

template <typename T>
template <typename E>
void MatrixCsr<T>::compress(const MatrixExpression<T, E>& other, T epsilon) {
    size_t numRows = other.rows();
    size_t numCols = other.cols();

    _size = {numRows, numCols};
    _nonZeros.clear();
    _columnIndices.clear();

    const E& expression = other();

    for (size_t i = 0; i < numRows; ++i) {
        _rowPointers.push_back(_nonZeros.size());

        for (size_t j = 0; j < numCols; ++j) {
            T val = expression(i, j);
            if (std::fabs(val) > epsilon) {
                _nonZeros.push_back(val);
                _columnIndices.push_back(j);
            }
        }
    }

    _rowPointers.push_back(_nonZeros.size());
}

template <typename T>
void MatrixCsr<T>::addElement(size_t i, size_t j, const T& value) {
    addElement({i, j, value});
}

template <typename T>
void MatrixCsr<T>::addElement(const Element& element) {
    ssize_t numRowsToAdd = (ssize_t)element.i - (ssize_t)_size.x + 1;
    if (numRowsToAdd > 0) {
        for (ssize_t i = 0; i < numRowsToAdd; ++i) {
            addRow({}, {});
        }
    }

    _size.y = std::max(_size.y, element.j + 1);

    size_t rowBegin = _rowPointers[element.i];
    size_t rowEnd = _rowPointers[element.i + 1];

    auto colIdxIter =
        std::lower_bound(_columnIndices.begin() + rowBegin,
                         _columnIndices.begin() + rowEnd, element.j);
    auto offset = colIdxIter - _columnIndices.begin();

    _columnIndices.insert(colIdxIter, element.j);
    _nonZeros.insert(_nonZeros.begin() + offset, element.value);

    for (size_t i = element.i + 1; i < _rowPointers.size(); ++i) {
        ++_rowPointers[i];
    }
}

template <typename T>
void MatrixCsr<T>::addRow(const NonZeroContainerType& nonZeros,
                          const IndexContainerType& columnIndices) {
    JET_ASSERT(nonZeros.size() == columnIndices.size());

    ++_size.x;

    // TODO: Implement zip iterator
    std::vector<std::pair<T, size_t>> zipped;
    for (size_t i = 0; i < nonZeros.size(); ++i) {
        zipped.emplace_back(nonZeros[i], columnIndices[i]);
        _size.y = std::max(_size.y, columnIndices[i] + 1);
    }
    std::sort(zipped.begin(), zipped.end(),
              [](std::pair<T, size_t> a, std::pair<T, size_t> b) {
                  return a.second < b.second;
              });
    for (size_t i = 0; i < zipped.size(); ++i) {
        _nonZeros.push_back(zipped[i].first);
        _columnIndices.push_back(zipped[i].second);
    }

    _rowPointers.push_back(_nonZeros.size());
}

template <typename T>
void MatrixCsr<T>::setElement(size_t i, size_t j, const T& value) {
    setElement({i, j, value});
}

template <typename T>
void MatrixCsr<T>::setElement(const Element& element) {
    size_t nzIndex = hasElement(element.i, element.j);
    if (nzIndex == kMaxSize) {
        addElement(element);
    } else {
        _nonZeros[nzIndex] = element.value;
    }
}

template <typename T>
bool MatrixCsr<T>::isEqual(const MatrixCsr& other) const {
    if (_size != other._size) {
        return false;
    }

    if (_nonZeros.size() != other._nonZeros.size()) {
        return false;
    }

    for (size_t i = 0; i < _nonZeros.size(); ++i) {
        if (_nonZeros[i] != other._nonZeros[i]) {
            return false;
        }
        if (_columnIndices[i] != other._columnIndices[i]) {
            return false;
        }
    }

    for (size_t i = 0; i < _rowPointers.size(); ++i) {
        if (_rowPointers[i] != other._rowPointers[i]) {
            return false;
        }
    }

    return true;
}

template <typename T>
bool MatrixCsr<T>::isSimilar(const MatrixCsr& other, double tol) const {
    if (_size != other._size) {
        return false;
    }

    if (_nonZeros.size() != other._nonZeros.size()) {
        return false;
    }

    for (size_t i = 0; i < _nonZeros.size(); ++i) {
        if (std::fabs(_nonZeros[i] - other._nonZeros[i]) > tol) {
            return false;
        }
        if (_columnIndices[i] != other._columnIndices[i]) {
            return false;
        }
    }

    for (size_t i = 0; i < _rowPointers.size(); ++i) {
        if (_rowPointers[i] != other._rowPointers[i]) {
            return false;
        }
    }

    return true;
}

template <typename T>
bool MatrixCsr<T>::isSquare() const {
    return rows() == cols();
}

template <typename T>
Size2 MatrixCsr<T>::size() const {
    return _size;
}

template <typename T>
size_t MatrixCsr<T>::rows() const {
    return _size.x;
}

template <typename T>
size_t MatrixCsr<T>::cols() const {
    return _size.y;
}

template <typename T>
size_t MatrixCsr<T>::numberOfNonZeros() const {
    return _nonZeros.size();
}

template <typename T>
const T& MatrixCsr<T>::nonZero(size_t i) const {
    return _nonZeros[i];
}

template <typename T>
T& MatrixCsr<T>::nonZero(size_t i) {
    return _nonZeros[i];
}

template <typename T>
const size_t& MatrixCsr<T>::rowPointer(size_t i) const {
    return _rowPointers[i];
}

template <typename T>
const size_t& MatrixCsr<T>::columnIndex(size_t i) const {
    return _columnIndices[i];
}

template <typename T>
T* MatrixCsr<T>::nonZeroData() {
    return _nonZeros.data();
}

template <typename T>
const T* const MatrixCsr<T>::nonZeroData() const {
    return _nonZeros.data();
}

template <typename T>
const size_t* const MatrixCsr<T>::rowPointersData() const {
    return _rowPointers.data();
}

template <typename T>
const size_t* const MatrixCsr<T>::columnIndicesData() const {
    return _columnIndices.data();
}

template <typename T>
typename MatrixCsr<T>::NonZeroIterator MatrixCsr<T>::nonZeroBegin() {
    return _nonZeros.begin();
}

template <typename T>
typename MatrixCsr<T>::ConstNonZeroIterator MatrixCsr<T>::nonZeroBegin() const {
    return _nonZeros.cbegin();
}

template <typename T>
typename MatrixCsr<T>::NonZeroIterator MatrixCsr<T>::nonZeroEnd() {
    return _nonZeros.end();
}

template <typename T>
typename MatrixCsr<T>::ConstNonZeroIterator MatrixCsr<T>::nonZeroEnd() const {
    return _nonZeros.cend();
}

template <typename T>
typename MatrixCsr<T>::IndexIterator MatrixCsr<T>::rowPointersBegin() {
    return _rowPointers.begin();
}

template <typename T>
typename MatrixCsr<T>::ConstIndexIterator MatrixCsr<T>::rowPointersBegin()
    const {
    return _rowPointers.cbegin();
}

template <typename T>
typename MatrixCsr<T>::IndexIterator MatrixCsr<T>::rowPointersEnd() {
    return _rowPointers.end();
}

template <typename T>
typename MatrixCsr<T>::ConstIndexIterator MatrixCsr<T>::rowPointersEnd() const {
    return _rowPointers.cend();
}

template <typename T>
typename MatrixCsr<T>::IndexIterator MatrixCsr<T>::columnIndicesBegin() {
    return _columnIndices.begin();
}

template <typename T>
typename MatrixCsr<T>::ConstIndexIterator MatrixCsr<T>::columnIndicesBegin()
    const {
    return _columnIndices.cbegin();
}

template <typename T>
typename MatrixCsr<T>::IndexIterator MatrixCsr<T>::columnIndicesEnd() {
    return _columnIndices.end();
}

template <typename T>
typename MatrixCsr<T>::ConstIndexIterator MatrixCsr<T>::columnIndicesEnd()
    const {
    return _columnIndices.cend();
}

template <typename T>
MatrixCsr<T> MatrixCsr<T>::add(const T& s) const {
    MatrixCsr ret(*this);
    parallelFor(kZeroSize, ret._nonZeros.size(),
                [&](size_t i) { ret._nonZeros[i] += s; });
    return ret;
}

template <typename T>
MatrixCsr<T> MatrixCsr<T>::add(const MatrixCsr& m) const {
    return binaryOp(m, std::plus<T>());
}

template <typename T>
MatrixCsr<T> MatrixCsr<T>::sub(const T& s) const {
    MatrixCsr ret(*this);
    parallelFor(kZeroSize, ret._nonZeros.size(),
                [&](size_t i) { ret._nonZeros[i] -= s; });
    return ret;
}

template <typename T>
MatrixCsr<T> MatrixCsr<T>::sub(const MatrixCsr& m) const {
    return binaryOp(m, std::minus<T>());
}

template <typename T>
MatrixCsr<T> MatrixCsr<T>::mul(const T& s) const {
    MatrixCsr ret(*this);
    parallelFor(kZeroSize, ret._nonZeros.size(),
                [&](size_t i) { ret._nonZeros[i] *= s; });
    return ret;
}

template <typename T>
template <typename VE>
MatrixCsrVectorMul<T, VE> MatrixCsr<T>::mul(
    const VectorExpression<T, VE>& v) const {
    return MatrixCsrVectorMul<T, VE>(*this, v());
};

template <typename T>
template <typename ME>
MatrixCsrMatrixMul<T, ME> MatrixCsr<T>::mul(
    const MatrixExpression<T, ME>& m) const {
    return MatrixCsrMatrixMul<T, ME>(*this, m());
}

template <typename T>
MatrixCsr<T> MatrixCsr<T>::div(const T& s) const {
    MatrixCsr ret(*this);
    parallelFor(kZeroSize, ret._nonZeros.size(),
                [&](size_t i) { ret._nonZeros[i] /= s; });
    return ret;
}

template <typename T>
MatrixCsr<T> MatrixCsr<T>::radd(const T& s) const {
    return add(s);
}

template <typename T>
MatrixCsr<T> MatrixCsr<T>::radd(const MatrixCsr& m) const {
    return add(m);
}

template <typename T>
MatrixCsr<T> MatrixCsr<T>::rsub(const T& s) const {
    MatrixCsr ret(*this);
    parallelFor(kZeroSize, ret._nonZeros.size(),
                [&](size_t i) { ret._nonZeros[i] = s - ret._nonZeros[i]; });
    return ret;
}

template <typename T>
MatrixCsr<T> MatrixCsr<T>::rsub(const MatrixCsr& m) const {
    return m.sub(*this);
}

template <typename T>
MatrixCsr<T> MatrixCsr<T>::rmul(const T& s) const {
    return mul(s);
}

template <typename T>
MatrixCsr<T> MatrixCsr<T>::rdiv(const T& s) const {
    MatrixCsr ret(*this);
    parallelFor(kZeroSize, ret._nonZeros.size(),
                [&](size_t i) { ret._nonZeros[i] = s / ret._nonZeros[i]; });
    return ret;
}

template <typename T>
void MatrixCsr<T>::iadd(const T& s) {
    parallelFor(kZeroSize, _nonZeros.size(),
                [&](size_t i) { _nonZeros[i] += s; });
}

template <typename T>
void MatrixCsr<T>::iadd(const MatrixCsr& m) {
    set(add(m));
}

template <typename T>
void MatrixCsr<T>::isub(const T& s) {
    parallelFor(kZeroSize, _nonZeros.size(),
                [&](size_t i) { _nonZeros[i] -= s; });
}

template <typename T>
void MatrixCsr<T>::isub(const MatrixCsr& m) {
    set(sub(m));
}

template <typename T>
void MatrixCsr<T>::imul(const T& s) {
    parallelFor(kZeroSize, _nonZeros.size(),
                [&](size_t i) { _nonZeros[i] *= s; });
}

template <typename T>
template <typename ME>
void MatrixCsr<T>::imul(const MatrixExpression<T, ME>& m) {
    MatrixCsrD result = mul(m);
    *this = std::move(result);
}

template <typename T>
void MatrixCsr<T>::idiv(const T& s) {
    parallelFor(kZeroSize, _nonZeros.size(),
                [&](size_t i) { _nonZeros[i] /= s; });
}

template <typename T>
T MatrixCsr<T>::sum() const {
    return parallelReduce(kZeroSize, numberOfNonZeros(), T(0),
                          [&](size_t start, size_t end, T init) {
                              T result = init;
                              for (size_t i = start; i < end; ++i) {
                                  result += _nonZeros[i];
                              }
                              return result;
                          },
                          std::plus<T>());
}

template <typename T>
T MatrixCsr<T>::avg() const {
    return sum() / numberOfNonZeros();
}

template <typename T>
T MatrixCsr<T>::min() const {
    const T& (*_min)(const T&, const T&) = std::min<T>;
    return parallelReduce(kZeroSize, numberOfNonZeros(),
                          std::numeric_limits<T>::max(),
                          [&](size_t start, size_t end, T init) {
                              T result = init;
                              for (size_t i = start; i < end; ++i) {
                                  result = std::min(result, _nonZeros[i]);
                              }
                              return result;
                          },
                          _min);
}

template <typename T>
T MatrixCsr<T>::max() const {
    const T& (*_max)(const T&, const T&) = std::max<T>;
    return parallelReduce(kZeroSize, numberOfNonZeros(),
                          std::numeric_limits<T>::min(),
                          [&](size_t start, size_t end, T init) {
                              T result = init;
                              for (size_t i = start; i < end; ++i) {
                                  result = std::max(result, _nonZeros[i]);
                              }
                              return result;
                          },
                          _max);
}

template <typename T>
T MatrixCsr<T>::absmin() const {
    return parallelReduce(kZeroSize, numberOfNonZeros(),
                          std::numeric_limits<T>::max(),
                          [&](size_t start, size_t end, T init) {
                              T result = init;
                              for (size_t i = start; i < end; ++i) {
                                  result = jet::absmin(result, _nonZeros[i]);
                              }
                              return result;
                          },
                          jet::absmin<T>);
}

template <typename T>
T MatrixCsr<T>::absmax() const {
    return parallelReduce(kZeroSize, numberOfNonZeros(), T(0),
                          [&](size_t start, size_t end, T init) {
                              T result = init;
                              for (size_t i = start; i < end; ++i) {
                                  result = jet::absmax(result, _nonZeros[i]);
                              }
                              return result;
                          },
                          jet::absmax<T>);
}

template <typename T>
T MatrixCsr<T>::trace() const {
    JET_ASSERT(isSquare());
    return parallelReduce(kZeroSize, rows(), T(0),
                          [&](size_t start, size_t end, T init) {
                              T result = init;
                              for (size_t i = start; i < end; ++i) {
                                  result += (*this)(i, i);
                              }
                              return result;
                          },
                          std::plus<T>());
}

template <typename T>
template <typename U>
MatrixCsr<U> MatrixCsr<T>::castTo() const {
    MatrixCsr<U> ret;
    ret.reserve(rows(), cols(), numberOfNonZeros());

    auto nnz = ret.nonZeroBegin();
    auto ci = ret.columnIndicesBegin();
    auto rp = ret.rowPointersBegin();

    parallelFor(kZeroSize, _nonZeros.size(), [&](size_t i) {
        nnz[i] = static_cast<U>(_nonZeros[i]);
        ci[i] = _columnIndices[i];
    });

    parallelFor(kZeroSize, _rowPointers.size(),
                [&](size_t i) { rp[i] = _rowPointers[i]; });

    return ret;
}

template <typename T>
template <typename E>
MatrixCsr<T>& MatrixCsr<T>::operator=(const E& m) {
    set(m);
    return *this;
}

template <typename T>
MatrixCsr<T>& MatrixCsr<T>::operator=(const MatrixCsr& other) {
    set(other);
    return *this;
}

template <typename T>
MatrixCsr<T>& MatrixCsr<T>::operator=(MatrixCsr&& other) {
    _size = other._size;
    other._size = Size2();
    _nonZeros = std::move(other._nonZeros);
    _rowPointers = std::move(other._rowPointers);
    _columnIndices = std::move(other._columnIndices);
    return *this;
}

template <typename T>
MatrixCsr<T>& MatrixCsr<T>::operator+=(const T& s) {
    iadd(s);
    return *this;
}

template <typename T>
MatrixCsr<T>& MatrixCsr<T>::operator+=(const MatrixCsr& m) {
    iadd(m);
    return *this;
}

template <typename T>
MatrixCsr<T>& MatrixCsr<T>::operator-=(const T& s) {
    isub(s);
    return *this;
}

template <typename T>
MatrixCsr<T>& MatrixCsr<T>::operator-=(const MatrixCsr& m) {
    isub(m);
    return *this;
}

template <typename T>
MatrixCsr<T>& MatrixCsr<T>::operator*=(const T& s) {
    imul(s);
    return *this;
}

template <typename T>
template <typename ME>
MatrixCsr<T>& MatrixCsr<T>::operator*=(const MatrixExpression<T, ME>& m) {
    imul(m);
    return *this;
}

template <typename T>
MatrixCsr<T>& MatrixCsr<T>::operator/=(const T& s) {
    idiv(s);
    return *this;
}

template <typename T>
T MatrixCsr<T>::operator()(size_t i, size_t j) const {
    size_t nzIndex = hasElement(i, j);
    if (nzIndex == kMaxSize) {
        return 0.0;
    } else {
        return _nonZeros[nzIndex];
    }
}

template <typename T>
bool MatrixCsr<T>::operator==(const MatrixCsr& m) const {
    return isEqual(m);
}

template <typename T>
bool MatrixCsr<T>::operator!=(const MatrixCsr& m) const {
    return !isEqual(m);
}

template <typename T>
MatrixCsr<T> MatrixCsr<T>::makeIdentity(size_t m) {
    MatrixCsr ret;
    ret._size = Size2(m, m);
    ret._nonZeros.resize(m, 1.0);
    ret._columnIndices.resize(m);
    std::iota(ret._columnIndices.begin(), ret._columnIndices.end(), kZeroSize);
    ret._rowPointers.resize(m + 1);
    std::iota(ret._rowPointers.begin(), ret._rowPointers.end(), kZeroSize);
    return ret;
}

template <typename T>
size_t MatrixCsr<T>::hasElement(size_t i, size_t j) const {
    if (i >= _size.x || j >= _size.y) {
        return kMaxSize;
    }

    size_t rowBegin = _rowPointers[i];
    size_t rowEnd = _rowPointers[i + 1];

    auto iter = binaryFind(_columnIndices.begin() + rowBegin,
                           _columnIndices.begin() + rowEnd, j);
    if (iter != _columnIndices.begin() + rowEnd) {
        return static_cast<size_t>(iter - _columnIndices.begin());
    } else {
        return kMaxSize;
    }
}

template <typename T>
template <typename Op>
MatrixCsr<T> MatrixCsr<T>::binaryOp(const MatrixCsr& m, Op op) const {
    JET_ASSERT(_size == m._size);

    MatrixCsr ret;

    for (size_t i = 0; i < _size.x; ++i) {
        std::vector<size_t> col;
        std::vector<double> nnz;

        auto colIterA = _columnIndices.begin() + _rowPointers[i];
        auto colIterB = m._columnIndices.begin() + m._rowPointers[i];
        auto colEndA = _columnIndices.begin() + _rowPointers[i + 1];
        auto colEndB = m._columnIndices.begin() + m._rowPointers[i + 1];
        auto nnzIterA = _nonZeros.begin() + _rowPointers[i];
        auto nnzIterB = m._nonZeros.begin() + m._rowPointers[i];

        while (colIterA != colEndA || colIterB != colEndB) {
            if (colIterB == colEndB || *colIterA < *colIterB) {
                col.push_back(*colIterA);
                nnz.push_back(op(*nnzIterA, 0));
                ++colIterA;
                ++nnzIterA;
            } else if (colIterA == colEndA || *colIterA > *colIterB) {
                col.push_back(*colIterB);
                nnz.push_back(op(0, *nnzIterB));
                ++colIterB;
                ++nnzIterB;
            } else {
                JET_ASSERT(*colIterA == *colIterB);
                col.push_back(*colIterB);
                nnz.push_back(op(*nnzIterA, *nnzIterB));
                ++colIterA;
                ++nnzIterA;
                ++colIterB;
                ++nnzIterB;
            }
        }

        ret.addRow(nnz, col);
    }

    return ret;
}

// MARK: Operator overloadings

template <typename T>
MatrixCsr<T> operator-(const MatrixCsr<T>& a) {
    return a.mul(-1);
}

template <typename T>
MatrixCsr<T> operator+(const MatrixCsr<T>& a, const MatrixCsr<T>& b) {
    return a.add(b);
}

template <typename T>
MatrixCsr<T> operator+(const MatrixCsr<T>& a, T b) {
    return a.add(b);
}

template <typename T>
MatrixCsr<T> operator+(T a, const MatrixCsr<T>& b) {
    return b.add(a);
}

template <typename T>
MatrixCsr<T> operator-(const MatrixCsr<T>& a, const MatrixCsr<T>& b) {
    return a.sub(b);
}

template <typename T>
MatrixCsr<T> operator-(const MatrixCsr<T>& a, T b) {
    return a.sub(b);
}

template <typename T>
MatrixCsr<T> operator-(T a, const MatrixCsr<T>& b) {
    return b.rsub(a);
}

template <typename T>
MatrixCsr<T> operator*(const MatrixCsr<T>& a, T b) {
    return a.mul(b);
}

template <typename T>
MatrixCsr<T> operator*(T a, const MatrixCsr<T>& b) {
    return b.rmul(a);
}

template <typename T, typename VE>
MatrixCsrVectorMul<T, VE> operator*(const MatrixCsr<T>& a,
                                    const VectorExpression<T, VE>& b) {
    return a.mul(b);
}

template <typename T, typename ME>
MatrixCsrMatrixMul<T, ME> operator*(const MatrixCsr<T>& a,
                                    const MatrixExpression<T, ME>& b) {
    return a.mul(b);
}

template <typename T>
MatrixCsr<T> operator/(const MatrixCsr<T>& a, T b) {
    return a.div(b);
}

template <typename T>
MatrixCsr<T> operator/(T a, const MatrixCsr<T>& b) {
    return b.rdiv(a);
}

}  // namespace jet

#endif  // INCLUDE_JET_DETAIL_MATRIX_CSR_INL_H_
