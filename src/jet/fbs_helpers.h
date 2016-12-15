// Copyright (c) 2016 Doyub Kim

#ifndef SRC_JET_FBS_HELPERS_H_
#define SRC_JET_FBS_HELPERS_H_

#include <basic_types_generated.h>
#include <jet/size2.h>
#include <jet/size3.h>
#include <jet/vector2.h>
#include <jet/vector3.h>

namespace jet {

inline fbs::Size2 jetToFbs(const Size2& vec) {
    return fbs::Size2(vec.x, vec.y);
}

inline fbs::Size3 jetToFbs(const Size3& vec) {
    return fbs::Size3(vec.x, vec.y, vec.z);
}

inline fbs::Vector2D jetToFbs(const Vector2D& vec) {
    return fbs::Vector2D(vec.x, vec.y);
}

inline fbs::Vector3D jetToFbs(const Vector3D& vec) {
    return fbs::Vector3D(vec.x, vec.y, vec.z);
}

inline Size2 fbsToJet(const fbs::Size2& vec) {
    return Size2(vec.x(), vec.y());
}

inline Size3 fbsToJet(const fbs::Size3& vec) {
    return Size3(vec.x(), vec.y(), vec.z());
}

inline Vector2D fbsToJet(const fbs::Vector2D& vec) {
    return Vector2D(vec.x(), vec.y());
}

inline Vector3D fbsToJet(const fbs::Vector3D& vec) {
    return Vector3D(vec.x(), vec.y(), vec.z());
}

template <
    typename GridType,
    typename FbsFactoryFunc,
    typename FbsGridType>
void serializeGrid(
    flatbuffers::FlatBufferBuilder* builder,
    const std::vector<GridType>& gridList,
    FbsFactoryFunc func,
    std::vector<flatbuffers::Offset<FbsGridType>>* fbsGridList) {
    for (const auto& grid : gridList) {
        auto type = builder->CreateString(grid->gridTypeName());
        auto resolution = jetToFbs(grid->resolution());
        auto gridSpacing = jetToFbs(grid->gridSpacing());
        auto origin = jetToFbs(grid->origin());

        std::vector<double> gridData;
        grid->getData(&gridData);
        auto data = builder->CreateVector(gridData.data(), gridData.size());

        flatbuffers::Offset<FbsGridType> fbsGrid = func(
            *builder, type, &resolution, &gridSpacing, &origin, data);

        fbsGridList->push_back(fbsGrid);
    }
}

template <typename FbsGridList, typename GridType, typename FactoryFunc>
void deserializeGrid(
    FbsGridList* fbsGridList,
    FactoryFunc factoryFunc,
    std::vector<GridType>* gridList) {
    for (const auto& grid : (*fbsGridList)) {
        auto type = grid->type()->c_str();
        auto resolution = fbsToJet(*grid->resolution());
        auto gridSpacing = fbsToJet(*grid->gridSpacing());
        auto origin = fbsToJet(*grid->origin());
        auto data = grid->data();

        auto newGrid = factoryFunc(type);
        newGrid->resize(resolution, gridSpacing, origin);

        std::vector<double> gridData(data->size());
        std::copy(data->begin(), data->end(), gridData.begin());

        newGrid->setData(gridData);

        gridList->push_back(newGrid);
    }
}

}  // namespace jet

#endif  // SRC_JET_FBS_HELPERS_H_
