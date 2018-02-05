"""
Copyright (c) 2018 Doyub Kim

I am making my contributions/submissions to this project solely in my personal
capacity and am not conveying any rights to any intellectual property of any
third parties.
"""

import pyjet
import unittest


class MyAnimation(pyjet.Animation):
    def __init__(self):
        self.lastFrame = None
        super(MyAnimation, self).__init__()

    def onUpdate(self, frame):
        self.lastFrame = frame

class AnimationTests(unittest.TestCase):
    def testInheritance(self):
        anim = MyAnimation()
        f = pyjet.Frame(3, 0.02)
        anim.update(f)
        self.assertEqual(anim.lastFrame.index, 3)
        self.assertEqual(anim.lastFrame.timeIntervalInSeconds, 0.02)


def main():
    pyjet.Logging.mute()
    unittest.main()


if __name__ == '__main__':
    main()
