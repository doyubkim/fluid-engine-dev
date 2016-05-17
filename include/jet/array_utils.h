// Copyright (c) 2016 Doyub Kim

#ifndef INCLUDE_JET_ARRAY_UTILS_H_
#define INCLUDE_JET_ARRAY_UTILS_H_

#include <jet/array_accessor2.h>
#include <jet/array_accessor3.h>
#include <jet/array1.h>
#include <jet/array2.h>
#include <jet/array3.h>

namespace jet {

template <typename ArrayType, typename T>
void setRange1(
    size_t size,
    const T& value,
    ArrayType* output);

template <typename ArrayType, typename T>
void setRange1(
    size_t begin,
    size_t end,
    const T& value,
    ArrayType* output);

template <typename ArrayType, typename T>
void setRange1(
    ArrayType* output,
    size_t size,
    const T& value);

template <typename ArrayType1, typename ArrayType2>
void copyRange1(
    const ArrayType1& input,
    size_t size,
    ArrayType2* output);

template <typename ArrayType1, typename ArrayType2>
void copyRange1(
    const ArrayType1& input,
    size_t begin,
    size_t end,
    ArrayType2* output);

template <typename ArrayType1, typename ArrayType2>
void copyRange2(
    const ArrayType1& input,
    size_t sizeX,
    size_t sizeY,
    ArrayType2* output);

template <typename ArrayType1, typename ArrayType2>
void copyRange2(
    const ArrayType1& input,
    size_t beginX,
    size_t endX,
    size_t beginY,
    size_t endY,
    ArrayType2* output);

template <typename ArrayType1, typename ArrayType2>
void copyRange3(
    const ArrayType1& input,
    size_t sizeX,
    size_t sizeY,
    size_t sizeZ,
    ArrayType2* output);

template <typename ArrayType1, typename ArrayType2>
void copyRange3(
    const ArrayType1& input,
    size_t beginX,
    size_t endX,
    size_t beginY,
    size_t endY,
    size_t beginZ,
    size_t endZ,
    ArrayType2* output);

//!
//! Extrapolates 2-D input data from 'valid' (1) to 'invalid' (0) region.
//! This function iterates multiple times to propagate the 'valid' values
//! to nearby 'invalid' region. The maximum distance of the propagation is
//! equal to numberOfIterations. The input parameters 'valid' and 'data'
//! should be collocated.
//!
//! \param input - data to extrapolate
//! \param valid - set 1 if valid, else 0.
//! \param numberOfIterations - number of iterations for propagation
//! \param output - extrapolated output
//!
template <typename T>
void extrapolateToRegion(
    const ConstArrayAccessor2<T>& input,
    const ConstArrayAccessor2<char>& valid,
    unsigned int numberOfIterations,
    ArrayAccessor2<T> output);

//!
//! Extrapolates 2-D input data from 'valid' (1) to 'invalid' (0) region.
//! This function iterates multiple times to propagate the 'valid' values
//! to nearby 'invalid' region. The maximum distance of the propagation is
//! equal to numberOfIterations. The input parameters 'valid' and 'data'
//! should be collocated.
//!
//! \param input - data to extrapolate
//! \param valid - set 1 if valid, else 0.
//! \param numberOfIterations - number of iterations for propagation
//! \param output - extrapolated output
//!
template <typename T>
void extrapolateToRegion(
    const ConstArrayAccessor3<T>& input,
    const ConstArrayAccessor3<char>& valid,
    unsigned int numberOfIterations,
    ArrayAccessor3<T> output);

//!
//! Converts 2-D array to Comma Separated Value (CSV) stream.
//!
//! \param data - data to convert
//! \param strm - stream object to write CSV
//!
template <typename ArrayType>
void convertToCsv(const ArrayType& data, std::ostream* strm);

}  // namespace jet

#include "detail/array_utils-inl.h"

#endif  // INCLUDE_JET_ARRAY_UTILS_H_
