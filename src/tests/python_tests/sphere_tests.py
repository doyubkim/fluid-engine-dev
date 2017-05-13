"""
Copyright (c) 2017 Doyub Kim

I am making my contributions/submissions to this project solely in my personal
capacity and am not conveying any rights to any intellectual property of any
third parties.
"""

import pyjet
import unittest


class Sphere3Test(unittest.TestCase):
    def testInit(self):
        a = pyjet.Sphere3()
        self.assertEqual(a.isNormalFlipped, False)
        self.assertEqual(a.center, (0, 0, 0))
        self.assertEqual(a.radius, 1.0)


def main():
    pyjet.Logging.mute()
    unittest.main()


if __name__ == '__main__':
    main()
