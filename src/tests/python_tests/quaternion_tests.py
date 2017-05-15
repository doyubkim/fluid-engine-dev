"""
Copyright (c) 2017 Doyub Kim

I am making my contributions/submissions to this project solely in my personal
capacity and am not conveying any rights to any intellectual property of any
third parties.
"""

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


def main():
    pyjet.Logging.mute()
    unittest.main()


if __name__ == '__main__':
    main()
