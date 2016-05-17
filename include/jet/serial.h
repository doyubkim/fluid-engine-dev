// Copyright (c) 2016 Doyub Kim

#ifndef INCLUDE_JET_SERIAL_H_
#define INCLUDE_JET_SERIAL_H_

namespace jet {

template <typename RandomIterator, typename T>
void serialFill(
    const RandomIterator& begin, const RandomIterator& end, const T& value);

template <typename IndexType, typename Function>
void serialFor(
    IndexType beginIndex, IndexType endIndex, const Function& function);

template <typename IndexType, typename Function>
void serialFor(
    IndexType beginIndexX,
    IndexType endIndexX,
    IndexType beginIndexY,
    IndexType endIndexY,
    const Function& function);

template <typename IndexType, typename Function>
void serialFor(
    IndexType beginIndexX,
    IndexType endIndexX,
    IndexType beginIndexY,
    IndexType endIndexY,
    IndexType beginIndexZ,
    IndexType endIndexZ,
    const Function& function);

template<typename RandomIterator, typename SortingFunction>
void serialSort(const RandomIterator& begin, const RandomIterator& end);

template<typename RandomIterator, typename SortingFunction>
void serialSort(
    const RandomIterator& begin,
    const RandomIterator& end,
    const SortingFunction& sortingFunction);

}  // namespace jet

#include "detail/serial-inl.h"

#endif  // INCLUDE_JET_SERIAL_H_
