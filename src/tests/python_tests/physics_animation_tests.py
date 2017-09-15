"""
Copyright (c) 2017 Doyub Kim

I am making my contributions/submissions to this project solely in my personal
capacity and am not conveying any rights to any intellectual property of any
third parties.
"""

import math
import pyjet
import unittest


class MyPhysicsAnimation(pyjet.PhysicsAnimation):
    def __init__(self):
        self.init_data = 0
        self.adv_data = 0
        super(MyPhysicsAnimation, self).__init__()

    def onAdvanceTimeStep(self, timeIntervalInSeconds):
        self.adv_data += 1

    def numberOfSubTimeSteps(self, timeIntervalInSeconds):
        return int(math.ceil(timeIntervalInSeconds / 0.02))

    def onInitialize(self):
        self.init_data = 1


class PhysicsAnimationTests(unittest.TestCase):
    def testInheritance(self):
        anim = MyPhysicsAnimation()

        anim.isUsingFixedSubTimeSteps = False
        f = pyjet.Frame(index=3, timeIntervalInSeconds=0.1)
        anim.update(f)
        self.assertEqual(anim.init_data, 1)
        self.assertEqual(anim.adv_data, 20)


def main():
    pyjet.Logging.mute()
    unittest.main()


if __name__ == '__main__':
    main()
