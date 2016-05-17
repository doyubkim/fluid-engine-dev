#!/usr/bin/env python

"""
Renders manual_tests output files
"""

import render
import utils

if __name__ == '__main__':
    filenames = utils.get_all_files('manual_tests_output', ['*' + render.INPUT_ARRAY_FORMAT])
    filenames.sort()

    for filename in filenames:
        tags = render.parse_tags(filename)
        # Pick the first component only
        if render.Y_TAG in tags or render.Z_TAG in tags:
            continue
        # Pick the first frame only
        if render.is_animation(tags) and '0000' not in tags:
            continue
        if render.POINT2_TAG in tags:
            render.render_point2(filename)
        elif render.POINT3_TAG in tags:
            render.render_point3(filename)
        elif render.LINE2_TAG in tags:
            render.render_line2(filename)
        elif render.LINE3_TAG in tags:
            render.render_line3(filename)
        elif render.GRID2_TAG in tags:
            render.render_grid2(filename)
        elif render.GRID3_TAG in tags:
            render.render_grid3(filename)
