// Copyright (c) 2016 Doyub Kim

#ifndef INCLUDE_JET_PARALLEL_H_
#define INCLUDE_JET_PARALLEL_H_

namespace jet {

template <typename RandomIterator, typename T>
void parallelFill(
    const RandomIterator& begin,
    const RandomIterator& end,
    const T& value);

template <typename IndexType, typename Function>
void parallelFor(
    IndexType beginIndex,
    IndexType endIndex,
    const Function& function);

template <typename IndexType, typename Function>
void parallelFor(
    IndexType beginIndexX,
    IndexType endIndexX,
    IndexType beginIndexY,
    IndexType endIndexY,
    const Function& function);

template <typename IndexType, typename Function>
void parallelFor(
    IndexType beginIndexX,
    IndexType endIndexX,
    IndexType beginIndexY,
    IndexType endIndexY,
    IndexType beginIndexZ,
    IndexType endIndexZ,
    const Function& function);

template<typename RandomIterator>
void parallelSort(RandomIterator begin, RandomIterator end);

template<typename RandomIterator, typename CompareFunction>
void parallelSort(
    RandomIterator begin,
    RandomIterator end,
    CompareFunction compare);

}  // namespace jet

#include "detail/parallel-inl.h"

#endif  // INCLUDE_JET_PARALLEL_H_
