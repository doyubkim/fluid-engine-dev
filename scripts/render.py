"""
Copyright (c) 2016 Doyub Kim
"""

import fnmatch
import matplotlib.animation as animation
import matplotlib.pyplot as plt
import numpy as np
import os
import platform
import re
import utils

# Sample file name:
# my_data_#line2,x.npy
# my_data_#grid2,iso.npy
# my_data_#grid2,iso,0231.npy

INPUT_ARRAY_FORMAT = '.npy'
OUTPUT_BITMAT_FORMAT = '.png'
OUTPUT_VECTOR_FORMAT = '.pdf'
OUTPUT_MOVIE_FORMAT = '.mp4'

POINT2_TAG = 'point2'
POINT3_TAG = 'point3'
LINE2_TAG = 'line2'
LINE3_TAG = 'line3'
GRID2_TAG = 'grid2'
GRID3_TAG = 'grid3'
ISO_TAG = 'iso'
X_TAG = 'x'
Y_TAG = 'y'
Z_TAG = 'z'

video_writer = 'ffmpeg'
video_extra_args = ['-vcodec', 'libx264', '-pix_fmt', 'yuv420p']
if utils.is_linux():
    video_writer = 'mencoder'
    video_extra_args = []

def remove_ext(filename):
    return filename[:-len(INPUT_ARRAY_FORMAT)]

def parse_tags(filename):
    tokens = remove_ext(filename).split('#')
    if len(tokens) > 1:
        return tokens[1].split(',')
    else:
        return []

def is_animation(tags):
    for tag in tags:
        if fnmatch.filter([tag], '[0-9][0-9][0-9][0-9]'):
            return True
    return False

def get_output_bitmap_filename(filename):
    return remove_ext(filename) + OUTPUT_BITMAT_FORMAT

def get_output_movie_filename(filename):
    return remove_ext(filename.replace(',0000', '')) + OUTPUT_MOVIE_FORMAT

def render_still_points2(filename_x, filename_y, output_filename, **kwargs):
    has_xtick = True
    has_ytick = True
    markersize = 3
    color = 'k'
    if 'has_xtick' in kwargs:
        has_xtick = kwargs['has_xtick']
    if 'has_ytick' in kwargs:
        has_ytick = kwargs['has_ytick']
    if 'marker' in kwargs:
        marker = kwargs['marker']
    if 'color' in kwargs:
        color = kwargs['color']
    fig, ax = plt.subplots()
    if not has_xtick:
        ax.set_xticks(())
        ax.set_xticklabels(())
    if not has_ytick:
        ax.set_yticks(())
        ax.set_yticklabels(())
    ax.set_aspect('equal')
    if 'xlim' in kwargs:
        ax.set_xlim(kwargs['xlim'][0], kwargs['xlim'][1])
    if 'ylim' in kwargs:
        ax.set_ylim(kwargs['ylim'][0], kwargs['ylim'][1])
    data_x = np.load(filename_x)
    data_y = np.load(filename_y)
    plt.plot(data_x, data_y, 'bo', markersize=markersize, color=color)
    plt.savefig(output_filename)
    plt.close(fig)
    print 'Rendered <%s>' % output_filename

def render_point2(filename):
    if is_animation(parse_tags(filename)):
        data = []
        def get_pt_data(filename, frame):
            filename = filename.replace('0000', '%04d' % frame)
            data_x = np.load(filename)
            data_y = np.load(filename.replace(',' + X_TAG, ',' + Y_TAG))
            return (data_x, data_y)
        def update_pts(frame, pts):
            x, y = data[frame]
            pts.set_data(x, y)
            return pts,
        dirname = os.path.dirname(filename)
        basename = os.path.basename(filename).replace('0000', '[0-9][0-9][0-9][0-9]')
        seq = utils.get_all_files(dirname, [basename])
        seq.sort()
        if len(seq) == 0:
            return

        data_x, data_y = get_pt_data(seq[0], 0)
        has_bbox = False
        for frame in range(len(seq)):
            data_x, data_y = get_pt_data(seq[frame], frame)
            if len(data_x) > 0 and len(data_y) > 0:
                if has_bbox:
                    xmin = min(xmin, data_x.min())
                    xmax = max(xmax, data_x.max())
                    ymin = min(ymin, data_y.min())
                    ymax = max(ymax, data_y.max())
                else:
                    xmin = data_x.min()
                    xmax = data_x.max()
                    ymin = data_y.min()
                    ymax = data_y.max()
                    has_bbox = True
            data.append((data_x, data_y))

        fig, ax = plt.subplots()
        x, y = get_pt_data(filename, 0)

        if has_bbox:
            xmid = (xmax + xmin) / 2.0
            ymid = (ymax + ymin) / 2.0
            new_xmin = xmid - 1.25 * (xmid - xmin)
            new_xmax = xmid + 1.25 * (xmax - xmid)
            new_ymin = ymid - 1.25 * (ymid - ymin)
            new_ymax = ymid + 1.25 * (ymax - ymid)
            ax.set_xlim(new_xmin, new_xmax)
            ax.set_ylim(new_ymin, new_ymax)
        ax.set_aspect('equal')
        pt, = ax.plot(x, y, 'bo', markersize=3)
        anim = animation.FuncAnimation(fig, update_pts, len(seq), fargs=(pt,), interval=60, blit=False)
        output_filename = get_output_movie_filename(filename.replace(',' + X_TAG, ''))
        anim.save(output_filename, fps=60, bitrate=5000, writer=video_writer, extra_args=video_extra_args)
        plt.close(fig)
        print 'Rendered <%s>' % output_filename
    else:
        fig, ax = plt.subplots()
        ax.set_aspect('equal')
        data_x = np.load(filename)
        data_y = np.load(filename.replace(',' + X_TAG, ',' + Y_TAG))
        plt.plot(data_x, data_y, 'bo', markersize=3)
        output_filename = get_output_bitmap_filename(filename.replace(',' + X_TAG, ''))
        plt.savefig(output_filename)
        plt.close(fig)
        print 'Rendered <%s>' % output_filename

def render_point3(filename):
    print 'Rendering <%s> as point3' % filename

def render_still_line2(filename_x, filename_y, output_filename, **kwargs):
    has_xtick = True
    has_ytick = True
    marker = 'o'
    markersize = 3
    color = 'k'
    linestyle = '-'
    if 'has_xtick' in kwargs:
        has_xtick = kwargs['has_xtick']
    if 'has_ytick' in kwargs:
        has_ytick = kwargs['has_ytick']
    if 'marker' in kwargs:
        marker = kwargs['marker']
    if 'markersize' in kwargs:
        markersize = kwargs['markersize']
    if 'color' in kwargs:
        color = kwargs['color']
    if 'linestyle' in kwargs:
        linestyle = kwargs['linestyle']
    fig, ax = plt.subplots()
    if not has_xtick:
        ax.set_xticks(())
        ax.set_xticklabels(())
    if not has_ytick:
        ax.set_yticks(())
        ax.set_yticklabels(())
    ax.set_aspect('equal')
    if 'xlim' in kwargs:
        ax.set_xlim(kwargs['xlim'][0], kwargs['xlim'][1])
    if 'ylim' in kwargs:
        ax.set_ylim(kwargs['ylim'][0], kwargs['ylim'][1])
    data_x = np.load(filename_x)
    data_y = np.load(filename_y)
    plt.plot(data_x, data_y, linestyle, marker=marker, markersize=markersize, color=color)
    plt.savefig(output_filename)
    plt.close(fig)
    print 'Rendered <%s>' % output_filename

def render_line2(filename):
    if is_animation(parse_tags(filename)):
        data = []
        def get_line_data(filename, frame):
            filename = filename.replace('0000', '%04d' % frame)
            data_x = np.load(filename)
            data_y = np.load(filename.replace(',' + X_TAG, ',' + Y_TAG))
            return (data_x, data_y)
        def update_lines(frame, line):
            x, y = data[frame]
            line.set_data(x, y)
            return line,
        dirname = os.path.dirname(filename)
        basename = os.path.basename(filename).replace('0000', '[0-9][0-9][0-9][0-9]')
        seq = utils.get_all_files(dirname, [basename])
        seq.sort()
        if len(seq) == 0:
            return

        data_x, data_y = get_line_data(seq[0], 0)
        has_bbox = False
        for frame in range(len(seq)):
            data_x, data_y = get_line_data(seq[frame], frame)
            if len(data_x) > 0 and len(data_y) > 0:
                if has_bbox:
                    xmin = min(xmin, data_x.min())
                    xmax = max(xmax, data_x.max())
                    ymin = min(ymin, data_y.min())
                    ymax = max(ymax, data_y.max())
                else:
                    xmin = data_x.min()
                    xmax = data_x.max()
                    ymin = data_y.min()
                    ymax = data_y.max()
                    has_bbox = True
            data.append((data_x, data_y))

        fig, ax = plt.subplots()
        x, y = get_line_data(filename, 0)

        if has_bbox:
            xmid = (xmax + xmin) / 2.0
            ymid = (ymax + ymin) / 2.0
            new_xmin = xmid - 1.25 * (xmid - xmin)
            new_xmax = xmid + 1.25 * (xmax - xmid)
            new_ymin = ymid - 1.25 * (ymid - ymin)
            new_ymax = ymid + 1.25 * (ymax - ymid)
            ax.set_xlim(new_xmin, new_xmax)
            ax.set_ylim(new_ymin, new_ymax)
        ax.set_aspect('equal')
        line, = ax.plot(x, y, lw=2, marker='o', markersize=3)
        anim = animation.FuncAnimation(fig, update_lines, len(seq), fargs=(line,), interval=60, blit=False)
        output_filename = get_output_movie_filename(filename.replace(',' + X_TAG, ''))
        anim.save(output_filename, fps=60, bitrate=5000, writer=video_writer, extra_args=video_extra_args)
        plt.close(fig)
        print 'Rendered <%s>' % output_filename
    else:
        output_filename = get_output_bitmap_filename(filename.replace(',' + X_TAG, ''))
        render_still_line2(filename, filename.replace(',' + X_TAG, ',' + Y_TAG), output_filename)

def render_line3(filename):
    print 'Rendering <%s> as line3' % filename

def render_still_scalar_grid2(filename, output_filename, **kwargs):
    interpolation = 'nearest'
    has_iso = False
    has_colorbar = True
    has_xtick = True
    has_ytick = True
    has_iso_colors = True
    iso_colors='k'
    if 'interpolation' in kwargs:
        interpolation = kwargs['interpolation']
    if 'has_iso' in kwargs:
        has_iso = kwargs['has_iso']
    if 'has_colorbar' in kwargs:
        has_colorbar = kwargs['has_colorbar']
    if 'has_xtick' in kwargs:
        has_xtick = kwargs['has_xtick']
    if 'has_ytick' in kwargs:
        has_ytick = kwargs['has_ytick']
    if 'iso_colors' in kwargs:
        iso_colors = kwargs['iso_colors']
        has_iso_colors = True
    grid_data = np.load(filename)
    grid_data = np.flipud(grid_data)
    fig, ax = plt.subplots()
    if not has_xtick:
        ax.set_xticks(())
        ax.set_xticklabels(())
    if not has_ytick:
        ax.set_yticks(())
        ax.set_yticklabels(())
    im = ax.imshow(grid_data, cmap=plt.cm.gray, interpolation=interpolation)
    if has_iso:
        if has_iso_colors:
            plt.contour(grid_data, 10, colors=iso_colors)
        else:
            plt.contour(grid_data, 10)
    if has_colorbar:
        plt.colorbar(im)
    plt.savefig(output_filename)
    plt.close(fig)
    print 'Rendered <%s>' % output_filename

def render_still_vector_grid2(filename_x, filename_y, output_filename, **kwargs):
    has_xtick = True
    has_ytick = True
    if 'has_xtick' in kwargs:
        has_xtick = kwargs['has_xtick']
    if 'has_ytick' in kwargs:
        has_ytick = kwargs['has_ytick']
    grid_data_u = np.load(filename_x)
    grid_data_v = np.load(filename_y)
    nx = len(grid_data_u[0])
    ny = len(grid_data_u)
    X, Y = np.meshgrid(np.arange(0, 1, 1.0 / nx), np.arange(0, float(ny)/nx, 1.0 / nx))
    U = grid_data_u
    V = grid_data_v
    fig, ax = plt.subplots()
    if not has_xtick:
        ax.set_xticks(())
        ax.set_xticklabels(())
    if not has_ytick:
        ax.set_yticks(())
        ax.set_yticklabels(())
    ax.set_aspect('equal')
    plt.quiver(X, Y, U, V)
    plt.savefig(output_filename)
    plt.close(fig)
    print 'Rendered <%s>' % output_filename

def render_grid2(filename):
    if is_animation(parse_tags(filename)):
        dirname = os.path.dirname(filename)
        basename = os.path.basename(filename).replace('0000', '[0-9][0-9][0-9][0-9]')
        seq = utils.get_all_files(dirname, [basename])
        seq.sort()
        if len(seq) == 0:
            return
        grid_data = np.load(seq[0])
        grid_data = np.flipud(grid_data)
        fig, ax = plt.subplots()
        im = ax.imshow(grid_data, cmap=plt.cm.gray, interpolation='nearest')
        if ISO_TAG in parse_tags(filename):
            plt.contour(grid_data)

        def update_image(i):
            grid_data = np.load(seq[i])
            grid_data = np.flipud(grid_data)
            im.set_array(grid_data)
            if ISO_TAG in parse_tags(filename):
                plt.contour(grid_data)
            return im

        output_filename = get_output_movie_filename(filename.replace(',' + X_TAG, ''))
        anim = animation.FuncAnimation(fig, update_image, frames=len(seq), interval=60, blit=False)
        anim.save(output_filename, fps=60, bitrate=5000, writer=video_writer, extra_args=video_extra_args)
        plt.close(fig)
        print 'Rendered <%s>' % output_filename
    else:
        tags = parse_tags(filename)
        output_filename = get_output_bitmap_filename(filename)
        if X_TAG in tags:
            grid_data_u = np.load(filename)
            grid_data_v = np.load(filename.replace(X_TAG, Y_TAG))
            render_still_vector_grid2(filename, filename.replace(X_TAG, Y_TAG), output_filename)
        else:
            has_iso = ISO_TAG in parse_tags(filename)
            render_still_scalar_grid2(filename, output_filename, has_iso=has_iso)

def render_grid3(filename):
    print 'Rendering <%s> as grid3' % filename

