"""
Copyright (c) 2018 Doyub Kim

I am making my contributions/submissions to this project solely in my personal
capacity and am not conveying any rights to any intellectual property of any
third parties.
"""

import pyjet
import unittest


class VertexCenteredScalarGrid2Tests(unittest.TestCase):
    def testConstructor(self):
        a = pyjet.VertexCenteredScalarGrid2()
        self.assertEqual(a.resolution.x, 1)
        self.assertEqual(a.resolution.y, 1)


def main():
    pyjet.Logging.mute()
    unittest.main()


if __name__ == '__main__':
    main()
