"""
Copyright (c) 2016 Doyub Kim
"""

import os, sys
root_dir = os.path.dirname(File('SConstruct').rfile().abspath)
sys.path.append(os.path.join(root_dir, 'scripts'))
import header_gen
import json
import utils

Export('os', 'sys', 'utils')

if utils.is_windows():
    print 'Use Visual Studio for Windows build.'
    quit()

# Fire up all cylinders
if GetOption('num_jobs') <= 1:
    SetOption('num_jobs', utils.detect_num_cpus())

# Configure the envinronment
env = Environment(ENV = os.environ, tools=['default'])

AddOption(
    '--cfg',
    dest='config',
    type='string',
    nargs=1,
    action='store',
    help='Specify build config file')
config_filename = GetOption('config')
if not config_filename:
    if utils.is_mac():
        config_filename = 'build/config-osx-x86_64.json'
    else:
        config_filename = 'build/config-linux-x86_64.json'

if 'CC' in os.environ:
    env['CC'] = os.environ['CC']
if 'CXX' in os.environ:
    env['CXX'] = os.environ['CXX']

with open(config_filename, 'r') as config_file:
    config = json.load(config_file)
    if 'BUILDDIR' in config:
        env['BUILDDIR'] = '#' + config['BUILDDIR']
    else:
        env['BUILDDIR'] = '#obj'
    if 'CC' in config and 'CC' not in os.environ:
        env['CC'] = config['CC']
    if 'CXX' in config and 'CXX' not in os.environ:
        env['CXX'] = config['CXX']
    if 'CXXFLAGS' in config:
        env.Append(CXXFLAGS=config['CXXFLAGS'])
    if 'CPPDEFINES' in config:
        env.Append(CPPDEFINES=config['CPPDEFINES'])
    if 'CPPPATH' in config:
        env.Append(CPPPATH=config['CPPPATH'])
    if 'LIBS' in config:
        env.Append(LIBS=config['LIBS'])
    if 'LINKFLAGS' in config:
        env.Append(LINKFLAGS=config['LINKFLAGS'])
    if 'LIBPATH' in config:
        env.Append(LIBPATH=config['LIBPATH'])

Export('env')

# Configure install location
AddOption(
    '--dist',
    dest='dist',
    type='string',
    nargs=1,
    action='store',
    help='Manually specify the install location')

def build(script_file, exports = [], duplicate = 0):
    dir_name = os.path.dirname(script_file)
    return SConscript(
        script_file,
        exports,
        variant_dir = os.path.join(env['BUILDDIR'], dir_name), duplicate=duplicate)

def build_app(subdir, name, dependencies = []):
    app_env, app = build(os.path.join('src', subdir, name, 'SConscript'))
    Requires(app, jet)
    for dep in dependencies:
        Requires(app, os.path.join(env['BUILDDIR'], dep))
    env.Alias(name, app)

    option = '%s_args' % name
    AddOption('--' + option, dest=option, type='string', nargs=1, action='store', help='Arguments to be passed to %s.' % name)
    args = GetOption(option)
    args = '' if not args else args

    if 'run_' + name in COMMAND_LINE_TARGETS:
        action = os.path.join(env['BUILDDIR'][1:], os.path.join('src', subdir, name, name)) + ' ' + args
        run_app_cmd = env.Command(target='run_' + name, source=None, action=action)
        Requires(run_app_cmd, app)
        env.Alias('run_' + name, run_app_cmd)

# Pre-build steps
header_gen.main()

# External libraries
cnpy_env, cnpy = build('external/src/cnpy/SConscript')
gtest_env, gtest = build('external/src/gtest/SConscript')
pystring_env, gtest = build('external/src/pystring/SConscript')
libobj_env, libobj = build('external/src/obj/SConscript')

# Core libraries
jet_env, jet = build('src/jet/SConscript')
Requires(jet, os.path.join(env['BUILDDIR'], 'external/src/obj'))

# Examples
build_app('examples', 'hello_fluid_sim')
build_app('examples', 'hybrid_liquid_sim', ['src/jet', 'external/src/pystring'])
build_app('examples', 'level_set_liquid_sim', ['src/jet', 'external/src/pystring'])
build_app('examples', 'obj2sdf', ['src/jet'])
build_app('examples', 'particles2obj', ['src/jet'])
build_app('examples', 'particles2xml', ['src/jet'])
build_app('examples', 'smoke_sim', ['src/jet', 'external/src/cnpy', 'external/src/pystring'])
build_app('examples', 'sph_sim', ['src/jet', 'external/src/cnpy', 'external/src/pystring'])

# Tests
build_app('tests', 'manual_tests', ['external/src/cnpy', 'external/src/gtest', 'src/jet', 'external/src/pystring'])
build_app('tests', 'unit_tests', ['external/src/gtest', 'external/src/jet'])
build_app('tests', 'perf_tests', ['external/src/gtest', 'external/src/jet'])

# Install
if 'install' in COMMAND_LINE_TARGETS:
    dist_dir = GetOption('dist')
    if dist_dir == None:
        dist_dir = env.GetBuildPath(env['DISTDIR'])
    else:
        dist_dir = '#' + dist_dir

    lib_inst = env.Install(os.path.join(dist_dir, 'lib'), jet)
    inc_inst = env.Install(dist_dir, ['include'])
    env.Depends(lib_inst, inc_inst)
    env.Depends(lib_inst, jet)
    env.Depends(inc_inst, jet)
    env.Alias("install", lib_inst)
