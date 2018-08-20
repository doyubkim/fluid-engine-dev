// Copyright (c) 2018 Doyub Kim
//
// I am making my contributions/submissions to this project solely in my
// personal capacity and am not conveying any rights to any intellectual
// property of any third parties.

#ifndef SRC_JET_FBS_HELPERS_H_
#define SRC_JET_FBS_HELPERS_H_

#include <generated/basic_types_generated.h>
#include <jet/matrix.h>
#include <algorithm>
#include <vector>

namespace jet {

inline fbs::Vector2UZ jetToFbs(const Vector2UZ& vec) {
    return fbs::Vector2UZ(vec.x, vec.y);
}

inline fbs::Vector3UZ jetToFbs(const Vector3UZ& vec) {
    return fbs::Vector3UZ(vec.x, vec.y, vec.z);
}

inline fbs::Vector2D jetToFbs(const Vector2D& vec) {
    return fbs::Vector2D(vec.x, vec.y);
}

inline fbs::Vector3D jetToFbs(const Vector3D& vec) {
    return fbs::Vector3D(vec.x, vec.y, vec.z);
}

inline Vector2UZ fbsToJet(const fbs::Vector2UZ& vec) {
    return Vector2UZ({vec.x(), vec.y()});
}

inline Vector3UZ fbsToJet(const fbs::Vector3UZ& vec) {
    return Vector3UZ({vec.x(), vec.y(), vec.z()});
}

inline Vector2D fbsToJet(const fbs::Vector2D& vec) {
    return Vector2D(vec.x(), vec.y());
}

inline Vector3D fbsToJet(const fbs::Vector3D& vec) {
    return Vector3D(vec.x(), vec.y(), vec.z());
}

template <typename GridType, typename FbsFactoryFunc, typename FbsGridType>
void serializeGrid(flatbuffers::FlatBufferBuilder* builder,
                   const std::vector<GridType>& gridList, FbsFactoryFunc func,
                   std::vector<flatbuffers::Offset<FbsGridType>>* fbsGridList) {
    for (const auto& grid : gridList) {
        auto type = builder->CreateString(grid->typeName());

        std::vector<uint8_t> gridSerialized;
        grid->serialize(&gridSerialized);
        auto fbsGrid = func(*builder, type,
                            builder->CreateVector(gridSerialized.data(),
                                                  gridSerialized.size()));
        fbsGridList->push_back(fbsGrid);
    }
}

template <typename FbsGridList, typename GridType, typename FactoryFunc>
void deserializeGrid(FbsGridList* fbsGridList, FactoryFunc factoryFunc,
                     std::vector<GridType>* gridList) {
    for (const auto& grid : (*fbsGridList)) {
        auto type = grid->type()->c_str();

        std::vector<uint8_t> gridSerialized(grid->data()->begin(),
                                            grid->data()->end());

        auto newGrid = factoryFunc(type);
        newGrid->deserialize(gridSerialized);

        gridList->push_back(newGrid);
    }
}

}  // namespace jet

#endif  // SRC_JET_FBS_HELPERS_H_
