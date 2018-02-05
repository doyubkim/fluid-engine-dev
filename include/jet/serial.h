// Copyright (c) 2018 Doyub Kim
//
// I am making my contributions/submissions to this project solely in my
// personal capacity and am not conveying any rights to any intellectual
// property of any third parties.

#ifndef INCLUDE_JET_SERIAL_H_
#define INCLUDE_JET_SERIAL_H_

namespace jet {

//!
//! \brief      Fills from \p begin to \p end with \p value.
//!
//! This function fills a container specified by begin and end iterators with
//! single thread. The order of the filling is deterministic.
//!
//! \param[in]  begin          The begin iterator of a container.
//! \param[in]  end            The end iterator of a container.
//! \param[in]  value          The value to fill a container.
//!
//! \tparam     RandomIterator Random iterator type.
//! \tparam     T              Value type of a container.
//!
template <typename RandomIterator, typename T>
void serialFill(
    const RandomIterator& begin, const RandomIterator& end, const T& value);

//!
//! \brief      Makes a for-loop from \p beginIndex \p to endIndex.
//!
//! This function makes a for-loop specified by begin and end indices with
//! single thread. The order of the visit is deterministic.
//!
//! \param[in]  beginIndex The begin index.
//! \param[in]  endIndex   The end index.
//! \param[in]  function   The function to call for each index.
//!
//! \tparam     IndexType  Index type.
//! \tparam     Function   Function type.
//!
template <typename IndexType, typename Function>
void serialFor(
    IndexType beginIndex, IndexType endIndex, const Function& function);

//!
//! \brief      Makes a 2D nested for-loop.
//!
//! This function makes a 2D nested for-loop specified by begin and end indices
//! for each dimension. X will be the inner-most loop while Y is the outer-most.
//! The order of the visit is deterministic.
//!
//! \param[in]  beginIndexX The begin index in X dimension.
//! \param[in]  endIndexX   The end index in X dimension.
//! \param[in]  beginIndexY The begin index in Y dimension.
//! \param[in]  endIndexY   The end index in Y dimension.
//! \param[in]  function    The function to call for each index (i, j).
//!
//! \tparam     IndexType  Index type.
//! \tparam     Function   Function type.
//!
template <typename IndexType, typename Function>
void serialFor(
    IndexType beginIndexX,
    IndexType endIndexX,
    IndexType beginIndexY,
    IndexType endIndexY,
    const Function& function);

//!
//! \brief      Makes a 3D nested for-loop.
//!
//! This function makes a 3D nested for-loop specified by begin and end indices
//! for each dimension. X will be the inner-most loop while Z is the outer-most.
//! The order of the visit is deterministic.
//!
//! \param[in]  beginIndexX The begin index in X dimension.
//! \param[in]  endIndexX   The end index in X dimension.
//! \param[in]  beginIndexY The begin index in Y dimension.
//! \param[in]  endIndexY   The end index in Y dimension.
//! \param[in]  beginIndexZ The begin index in Z dimension.
//! \param[in]  endIndexZ   The end index in Z dimension.
//! \param[in]  function    The function to call for each index (i, j, k).
//!
//! \tparam     IndexType   Index type.
//! \tparam     Function    Function type.
//!
template <typename IndexType, typename Function>
void serialFor(
    IndexType beginIndexX,
    IndexType endIndexX,
    IndexType beginIndexY,
    IndexType endIndexY,
    IndexType beginIndexZ,
    IndexType endIndexZ,
    const Function& function);

//!
//! \brief      Sorts a container.
//!
//! This function sorts a container specified by begin and end iterators.
//!
//! \param[in]  begin          The begin random access iterator.
//! \param[in]  end            The end random access iterator.
//!
//! \tparam     RandomIterator Iterator type.
//!
template<typename RandomIterator>
void serialSort(RandomIterator begin, RandomIterator end);

//!
//! \brief      Sorts a container with a custom compare function.
//!
//! This function sorts a container specified by begin and end iterators. It
//! takes extra compare function which returns true if the first argument is
//! less than the second argument.
//!
//! \param[in]  begin           The begin random access iterator.
//! \param[in]  end             The end random access iterator.
//! \param[in]  compare         The compare function.
//!
//! \tparam     RandomIterator  Iterator type.
//! \tparam     CompareFunction Compare function type.
//!
template<typename RandomIterator, typename SortingFunction>
void serialSort(
    RandomIterator begin,
    RandomIterator end,
    const SortingFunction& sortingFunction);

}  // namespace jet

#include "detail/serial-inl.h"

#endif  // INCLUDE_JET_SERIAL_H_
