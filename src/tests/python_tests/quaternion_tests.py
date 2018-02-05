"""
Copyright (c) 2018 Doyub Kim

I am making my contributions/submissions to this project solely in my personal
capacity and am not conveying any rights to any intellectual property of any
third parties.
"""

import math
import pyjet
import unittest


class QuaternionTest(unittest.TestCase):
    def testInit(self):
        a = pyjet.QuaternionD()
        self.assertEqual(a.w, 1.0)
        self.assertEqual(a.x, 0.0)
        self.assertEqual(a.y, 0.0)
        self.assertEqual(a.z, 0.0)
        b = pyjet.QuaternionD(-1, 2, 3, 4)
        self.assertEqual(b.w, -1.0)
        self.assertEqual(b.x, 2.0)
        self.assertEqual(b.y, 3.0)
        self.assertEqual(b.z, 4.0)
        c = pyjet.QuaternionD(x=2, z=4, w=1, y=3)
        self.assertEqual(c.w, 1.0)
        self.assertEqual(c.x, 2.0)
        self.assertEqual(c.y, 3.0)
        self.assertEqual(c.z, 4.0)

    def testGetters(self):
        a = pyjet.QuaternionD(1, 2, 3, 4)
        self.assertEqual(a[0], a.w)
        self.assertEqual(a[1], a.x)
        self.assertEqual(a[2], a.y)
        self.assertEqual(a[3], a.z)
        a.normalize()
        axis = a.axis()
        angle = a.angle()
        denom = math.sqrt(1 - a.w * a.w)
        self.assertAlmostEqual(axis.x, math.sqrt(2.0 / 15.0) / denom)
        self.assertAlmostEqual(axis.y, math.sqrt(3.0 / 10.0) / denom)
        self.assertAlmostEqual(axis.z, 2.0 * math.sqrt(2.0 / 15.0) / denom)
        self.assertAlmostEqual(angle, 2.0 * math.acos(1.0 / math.sqrt(30.0)))

    def testSetters(self):
        a = pyjet.QuaternionD()
        a.setAxisAngle((0, -1, 0), math.pi / 2)
        axis = a.axis()
        angle = a.angle()
        self.assertEqual(axis.x, 0)
        self.assertEqual(axis.y, -1)
        self.assertEqual(axis.z, 0)
        self.assertEqual(angle, math.pi / 2)
        a[0] = 1.0; a[1] = 2.0;  a[2] = 3.0;  a[3] = 4.0
        self.assertEqual(1, a.w)
        self.assertEqual(2, a.x)
        self.assertEqual(3, a.y)
        self.assertEqual(4, a.z)


def main():
    pyjet.Logging.mute()
    unittest.main()


if __name__ == '__main__':
    main()
