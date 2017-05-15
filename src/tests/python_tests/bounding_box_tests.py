"""
Copyright (c) 2017 Doyub Kim

I am making my contributions/submissions to this project solely in my personal
capacity and am not conveying any rights to any intellectual property of any
third parties.
"""

import pyjet
import unittest


class BoundingBox3DTests(unittest.TestCase):
    def testInit(self):
        a = pyjet.BoundingBox3D()
        self.assertGreater(a.lowerCorner.x, a.upperCorner.x)
        self.assertGreater(a.lowerCorner.y, a.upperCorner.y)
        self.assertGreater(a.lowerCorner.z, a.upperCorner.z)
        b = pyjet.BoundingBox3D((-1, -2, -3), (4, 2, 5))
        self.assertEqual(b.lowerCorner.x, -1.0)
        self.assertEqual(b.lowerCorner.y, -2.0)
        self.assertEqual(b.lowerCorner.z, -3.0)
        self.assertEqual(b.upperCorner.x, 4.0)
        self.assertEqual(b.upperCorner.y, 2.0)
        self.assertEqual(b.upperCorner.z, 5.0)
        l, c = pyjet.Vector3D(-1, -2, -3), pyjet.Vector3D(4, 2, 5)
        c = pyjet.BoundingBox3D(l, c)
        self.assertEqual(c.lowerCorner.x, -1.0)
        self.assertEqual(c.lowerCorner.y, -2.0)
        self.assertEqual(c.lowerCorner.z, -3.0)
        self.assertEqual(c.upperCorner.x, 4.0)
        self.assertEqual(c.upperCorner.y, 2.0)
        self.assertEqual(c.upperCorner.z, 5.0)


def main():
    pyjet.Logging.mute()
    unittest.main()


if __name__ == '__main__':
    main()
