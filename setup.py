# Original code from https://github.com/pybind/cmake_example/blob/master/setup.py
#
# Copyright (c) 2019 Doyub Kim
#
# I am making my contributions/submissions to this project solely in my personal
# capacity and am not conveying any rights to any intellectual property of any
# third parties.
#

import os
import re
import sys
import platform
import subprocess
import multiprocessing

from setuptools import setup, Extension
from setuptools.command.build_ext import build_ext
from distutils.version import LooseVersion


class CMakeExtension(Extension):
    def __init__(self, name, sourcedir=''):
        Extension.__init__(self, name, sources=[])
        self.sourcedir = os.path.abspath(sourcedir)


class CMakeBuild(build_ext):
    def run(self):
        try:
            out = subprocess.check_output(['cmake', '--version'])
        except OSError:
            raise RuntimeError("CMake must be installed to build the following extensions: " +
                               ", ".join(e.name for e in self.extensions))

        if platform.system() == "Windows":
            cmake_version = LooseVersion(
                re.search(r'version\s*([\d.]+)', out.decode()).group(1))
            if cmake_version < '3.1.0':
                raise RuntimeError("CMake >= 3.1.0 is required on Windows")

        for ext in self.extensions:
            self.build_extension(ext)

    def build_extension(self, ext):
        extdir = os.path.abspath(os.path.dirname(
            self.get_ext_fullpath(ext.name)))
        cmake_args = ['-DCMAKE_LIBRARY_OUTPUT_DIRECTORY=' + extdir,
                      '-DPYTHON_EXECUTABLE=' + sys.executable,
                      '-DBUILD_FROM_PIP=ON']

        env = os.environ.copy()
        tasking_sys = env.get('TASKING_SYSTEM', '')
        if tasking_sys:
            cmake_args += ['-DJET_TASKING_SYSTEM=' + tasking_sys]

        cfg = 'Debug' if self.debug else 'Release'
        build_args = ['--config', cfg]

        if platform.system() == "Windows":
            cmake_args += [
                '-DCMAKE_LIBRARY_OUTPUT_DIRECTORY_{}={}'.format(cfg.upper(), extdir)]
            if sys.maxsize > 2**32:
                cmake_args += ['-A', 'x64']
            build_args += ['--', '/m']
        else:
            cmake_args += ['-DCMAKE_BUILD_TYPE=' + cfg]
            num_jobs = env.get('NUM_JOBS', multiprocessing.cpu_count())
            build_args += ['--', '-j%s' % str(num_jobs), 'pyjet']

        env['CXXFLAGS'] = '{} -DVERSION_INFO=\\"{}\\"'.format(env.get('CXXFLAGS', ''),
                                                              self.distribution.get_version())
        if not os.path.exists(self.build_temp):
            os.makedirs(self.build_temp)
        subprocess.check_call(['cmake', ext.sourcedir] +
                              cmake_args, cwd=self.build_temp, env=env)
        subprocess.check_call(['cmake', '--build', '.'] +
                              build_args, cwd=self.build_temp)


setup(
    name='pyjet',
    version='1.3.3',
    author='Doyub Kim',
    author_email='doyubkim@gmail.com',
    description='Fluid simulation engine for computer graphics applications',
    long_description='',
    ext_modules=[CMakeExtension('pyjet')],
    cmdclass=dict(build_ext=CMakeBuild),
    zip_safe=False,
)
