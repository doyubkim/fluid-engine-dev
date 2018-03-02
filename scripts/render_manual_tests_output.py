#!/usr/bin/env python

"""
Renders manual_tests output files
"""

import render
import utils
import argparse

def processArguments():
    parser = argparse.ArgumentParser(
        description='Renders matplotlib files generated with jet.\n'
                    'Run this utility from the same level as the manual_tests_output directory.')

    parser.add_argument('-f', '--fps', type=int, help='override animation frame rate (default is 60fps)')
    return parser.parse_args()

def getFrameRate(arguments):
    if arguments.fps:
        fps = arguments.fps
    else:
        fps = 60
    return fps

if __name__ == '__main__':

    fps = getFrameRate(processArguments())

    filenames = utils.get_all_files('manual_tests_output', ['*' + render.INPUT_ARRAY_FORMAT])
    filenames.sort()

    for filename in filenames:
        try:
            tags = render.parse_tags(filename)
            # Pick the first component only
            if render.Y_TAG in tags or render.Z_TAG in tags:
                continue
            # Pick the first frame only
            if render.is_animation(tags) and '0000' not in tags:
                continue
            if render.POINT2_TAG in tags:
                render.render_point2(filename, fps)
            elif render.POINT3_TAG in tags:
                render.render_point3(filename)
            elif render.LINE2_TAG in tags:
                render.render_line2(filename, fps)
            elif render.LINE3_TAG in tags:
                render.render_line3(filename)
            elif render.GRID2_TAG in tags:
                render.render_grid2(filename, fps)
            elif render.GRID3_TAG in tags:
                render.render_grid3(filename)
        except Exception as e:
            print ('Failed to render', filename)
            print ('Why?')
            print (e)


