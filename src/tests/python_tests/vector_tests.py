"""
Copyright (c) 2017 Doyub Kim

I am making my contributions/submissions to this project solely in my personal
capacity and am not conveying any rights to any intellectual property of any
third parties.
"""

import pyjet
import unittest


class Vector3DTest(unittest.TestCase):
    def testInit(self):
        a = pyjet.Vector3D()
        self.assertEqual(a.x, 0.0)
        self.assertEqual(a.y, 0.0)
        self.assertEqual(a.z, 0.0)
        b = pyjet.Vector3D(1, 2, 3)
        self.assertEqual(b.x, 1.0)
        self.assertEqual(b.y, 2.0)
        self.assertEqual(b.z, 3.0)
        c = pyjet.Vector3D(y=2, x=1, z=3)
        self.assertEqual(c.x, 1.0)
        self.assertEqual(c.y, 2.0)
        self.assertEqual(c.z, 3.0)

    def testCalc(self):
        a = pyjet.Vector3D(1, 2, 3)
        b = pyjet.Vector3D(4, 6, 8)
        c = a + b
        self.assertEqual(c.x, 5.0)
        self.assertEqual(c.y, 8.0)
        self.assertEqual(c.z, 11.0)
        c = a - b
        self.assertEqual(c.x, -3.0)
        self.assertEqual(c.y, -4.0)
        self.assertEqual(c.z, -5.0)
        c = a * b
        self.assertEqual(c.x, 4.0)
        self.assertEqual(c.y, 12.0)
        self.assertEqual(c.z, 24.0)
        c = a / b
        self.assertEqual(c.x, 1.0 / 4.0)
        self.assertEqual(c.y, 1.0 / 3.0)
        self.assertEqual(c.z, 3.0 / 8.0)


def main():
    pyjet.Logging.mute()
    unittest.main()


if __name__ == '__main__':
    main()
