// Copyright (c) 2018 Doyub Kim
//
// I am making my contributions/submissions to this project solely in my
// personal capacity and am not conveying any rights to any intellectual
// property of any third parties.

#include "manual_tests.h"

#include <jet/array1.h>
#include <jet/constant_vector_field3.h>
#include <jet/physics_animation.h>
#include <jet/vector3.h>

using namespace jet;

class SimpleMassSpringAnimation : public PhysicsAnimation
{
public:
    struct Edge
    {
        size_t first;
        size_t second;
    };

    struct Constraint
    {
        size_t pointIndex;
        Vector3D fixedPosition;
        Vector3D fixedVelocity;
    };

    std::vector<Vector3D> positions;
    std::vector<Vector3D> velocities;
    std::vector<Vector3D> forces;
    std::vector<Edge> edges;

    double mass = 1.0;
    Vector3D gravity = Vector3D(0.0, -9.8, 0.0);
    double stiffness = 500.0;
    double restLength = 1.0;
    double dampingCoefficient = 1.0;
    double dragCoefficient = 0.1;

    double floorPositionY = -7.0;
    double restitutionCoefficient = 0.3;

    VectorField3Ptr wind;

    std::vector<Constraint> constraints;

    SimpleMassSpringAnimation() {}

    void makeChain(size_t numberOfPoints)
    {
        if (numberOfPoints == 0)
        {
            return;
        }

        size_t numberOfEdges = numberOfPoints - 1;

        positions.resize(numberOfPoints);
        velocities.resize(numberOfPoints);
        forces.resize(numberOfPoints);
        edges.resize(numberOfEdges);

        for (size_t i = 0; i < numberOfPoints; ++i)
        {
            positions[i].x = -static_cast<double>(i);
        }

        for (size_t i = 0; i < numberOfEdges; ++i)
        {
            edges[i] = Edge{i, i + 1};
        }
    }

    void exportStates(Array1<double>& x, Array1<double>& y) const
    {
        x.resize(positions.size());
        y.resize(positions.size());

        for (size_t i = 0; i < positions.size(); ++i)
        {
            x[i] = positions[i].x;
            y[i] = positions[i].y;
        }
    }

protected:
    void onAdvanceTimeStep(double timeIntervalInSeconds) override
    {
        size_t numberOfPoints = positions.size();
        size_t numberOfEdges = edges.size();

        // Compute forces
        for (size_t i = 0; i < numberOfPoints; ++i)
        {
            // Gravity force
            forces[i] = mass * gravity;

            // Air drag force
            Vector3D relativeVel = velocities[i];
            if (wind != nullptr)
            {
                relativeVel -= wind->sample(positions[i]);
            }
            forces[i] += -dragCoefficient * relativeVel;
        }

        for (size_t i = 0; i < numberOfEdges; ++i)
        {
            size_t pointIndex0 = edges[i].first;
            size_t pointIndex1 = edges[i].second;

            // Compute spring force
            Vector3D pos0 = positions[pointIndex0];
            Vector3D pos1 = positions[pointIndex1];
            Vector3D r = pos0 - pos1;
            double distance = r.length();
            if (distance > 0.0)
            {
                Vector3D force = -stiffness * (distance - restLength) * r.normalized();
                forces[pointIndex0] += force;
                forces[pointIndex1] -= force;
            }

            // Add damping force
            Vector3D vel0 = velocities[pointIndex0];
            Vector3D vel1 = velocities[pointIndex1];
            Vector3D relativeVel0 = vel0 - vel1;
            Vector3D damping = -dampingCoefficient * relativeVel0;
            forces[pointIndex0] += damping;
            forces[pointIndex1] -= damping;
        }

        // Update states
        for (size_t i = 0; i < numberOfPoints; ++i)
        {
            // Compute new states
            Vector3D newAcceleration = forces[i] / mass;
            Vector3D newVelocity = velocities[i] + timeIntervalInSeconds * newAcceleration;
            Vector3D newPosition = positions[i] + timeIntervalInSeconds * newVelocity;

            // Collision
            if (newPosition.y < floorPositionY)
            {
                newPosition.y = floorPositionY;

                if (newVelocity.y < 0.0)
                {
                    newVelocity.y *= -restitutionCoefficient;
                    newPosition.y += timeIntervalInSeconds * newVelocity.y;
                }
            }

            // Update states
            velocities[i] = newVelocity;
            positions[i] = newPosition;
        }

        // Apply constraints
        for (size_t i = 0; i < constraints.size(); ++i)
        {
            size_t pointIndex = constraints[i].pointIndex;
            positions[pointIndex] = constraints[i].fixedPosition;
            velocities[pointIndex] = constraints[i].fixedVelocity;
        }
    }
};

JET_TESTS(PhysicsAnimation);

JET_BEGIN_TEST_F(PhysicsAnimation, SimpleMassSpringAnimation)
{
    Array1<double> x;
    Array1<double> y;

    SimpleMassSpringAnimation anim;
    anim.makeChain(10);
    anim.wind = std::make_shared<ConstantVectorField3>(Vector3D(30.0, 0.0, 0.0));
    anim.constraints.push_back(SimpleMassSpringAnimation::Constraint{0, Vector3D(), Vector3D()});
    anim.exportStates(x, y);

    char filename[256];
    snprintf(filename, sizeof(filename), "data.#line2,0000,x.npy");
    saveData(x.constAccessor(), filename);
    snprintf(filename, sizeof(filename), "data.#line2,0000,y.npy");
    saveData(y.constAccessor(), filename);

    for (Frame frame(0, 1.0 / 60.0); frame.index < 360; frame.advance())
    {
        anim.update(frame);
        anim.exportStates(x, y);

        snprintf(filename, sizeof(filename), "data.#line2,%04d,x.npy", frame.index);
        saveData(x.constAccessor(), filename);
        snprintf(filename, sizeof(filename), "data.#line2,%04d,y.npy", frame.index);
        saveData(y.constAccessor(), filename);
    }
}
JET_END_TEST_F
