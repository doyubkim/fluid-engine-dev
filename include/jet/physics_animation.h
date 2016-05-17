// Copyright (c) 2016 Doyub Kim

#ifndef INCLUDE_JET_PHYSICS_ANIMATION_H_
#define INCLUDE_JET_PHYSICS_ANIMATION_H_

#include <jet/animation.h>

namespace jet {

class PhysicsAnimation : public Animation {
 public:
    PhysicsAnimation();

    virtual ~PhysicsAnimation();

    bool isUsingFixedSubTimeSteps() const;

    void setIsUsingFixedSubTimeSteps(bool isUsing);

    unsigned int numberOfFixedSubTimeSteps() const;

    void setNumberOfFixedSubTimeSteps(unsigned int numberOfSteps);

 protected:
    virtual void onAdvanceTimeStep(double timeIntervalInSeconds) = 0;

    virtual unsigned int numberOfSubTimeSteps(
        double timeIntervalInSeconds) const;

 private:
    bool _isUsingFixedSubTimeSteps = true;
    unsigned int _numberOfFixedSubTimeSteps = 1;

    void onUpdate(const Frame& frame) final;
};

typedef std::shared_ptr<PhysicsAnimation> PhysicsAnimationPtr;

}  // namespace jet

#endif  // INCLUDE_JET_PHYSICS_ANIMATION_H_

