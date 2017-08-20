"""
Copyright (c) 2017 Doyub Kim

I am making my contributions/submissions to this project solely in my personal
capacity and am not conveying any rights to any intellectual property of any
third parties.
"""

import pyjet
import unittest
from animation_tests import *
from bounding_box_tests import *
from face_centered_grid_tests import *
from flip_solver_tests import *
from particle_system_data_tests import *
from physics_animation_tests import *
from sph_system_data_tests import *
from sphere_tests import *
from vector_tests import *
from quaternion_tests import *


def main():
    pyjet.Logging.mute()
    unittest.main()


if __name__ == '__main__':
    main()
